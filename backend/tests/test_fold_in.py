"""Online ridge fold-in tests: fixed item vectors, fresh ratings, and API integration."""
import numpy as np
import pandas as pd
import pytest
from recommender import HybridRecommender


def fixture_model():
    movies = pd.DataFrame([
        {'id': 100 + j, 'movieId': j + 1, 'title': f'Movie {j}',
         'genres': 'Drama' if j % 2 else 'Science Fiction',
         'overview': f'The story of movie number {j} on an adventure'}
        for j in range(12)
    ] + [{'id': 999, 'movieId': None, 'title': 'Unlinked movie',
          'genres': 'Science Fiction', 'overview': 'A new space adventure'}])
    ratings = pd.DataFrame([
        {'userId': user, 'movieId': movie, 'rating': 0.5 + ((movie + user) % 10) / 2}
        for user in range(1, 8) for movie in range(1, 13)
    ])
    return HybridRecommender(factors=4).fit(movies, ratings)


def test_fold_in_regularized_least_squares_matches_normal_equations():
    model = fixture_model()
    history = {100 + j: (j % 5) + 1.0 for j in range(6)}
    folded = model.fold_in_user(history, regularization=2.0)
    assert folded['support'] == 6
    Q = model.item_factors[np.array([model.id_to_col[mid] for mid in history])].astype(float)
    centered = np.array(list(history.values())) - folded['mean']
    lam = 2.0 * np.trace(Q.T @ Q) / Q.shape[1]
    np.testing.assert_allclose(
        (Q.T @ Q + lam * np.eye(Q.shape[1])) @ folded['factors'],
        Q.T @ centered, atol=1e-9,
    )
    assert np.all(np.isfinite(folded['factors']))


def test_fold_in_uses_fixed_factors_and_new_ratings():
    model = fixture_model()
    original_items = model.item_factors.copy()
    original_user_rows = len(model.user_to_row)
    history = {100 + j: float((j % 5) + 1) for j in range(7)}
    _, _, alpha, strategy = model._ranking_scores(additional_ratings=history, use_fold_in=True)
    assert strategy == 'fold_in_topk_hybrid'
    np.testing.assert_array_equal(alpha[model.item_counts > 0], 0.25)
    assert alpha[model.id_to_col[999]] == 0.0
    assert 0.0 <= float(alpha.min()) <= float(alpha.max()) <= 0.9
    history[106] = 1.5
    first = model.fold_in_user({**history, 106: 5})['factors']
    second = model.fold_in_user(history)['factors']
    assert not np.allclose(first, second)
    np.testing.assert_array_equal(model.item_factors, original_items)
    assert len(model.user_to_row) == original_user_rows


def test_fold_in_fallback_threshold_and_unlinked_movie():
    model = fixture_model()
    assert model.fold_in_user({100: 5.0, 101: 4.0, 999: 5.0}) is None
    history = {100 + j: 4.0 for j in range(4)}
    history[999] = 5.0
    assert model.fold_in_user(history) is None  # TMDB-only title does not count
    _, _, alpha, strategy = model._ranking_scores(additional_ratings=history, use_fold_in=True)
    assert strategy == 'topk_popularity'
    assert np.all(alpha == 0)
    history[104] = 3.5
    _, _, alpha, strategy = model._ranking_scores(additional_ratings=history, use_fold_in=True)
    assert strategy == 'fold_in_topk_hybrid'
    assert alpha[model.id_to_col[999]] == 0
    with pytest.raises(ValueError):
        model.fold_in_user(history, regularization=0)


def test_fold_in_weights_increase_with_history():
    model = fixture_model()
    assert model._fold_in_weight(5) == 0.25
    assert model._fold_in_weight(10) == 0.5
    assert model._fold_in_weight(20) == 0.9
    history = {100 + j: 4.0 for j in range(10)}
    _, _, alpha, strategy = model._ranking_scores(additional_ratings=history, use_fold_in=True)
    assert strategy == 'fold_in_topk_hybrid'
    np.testing.assert_array_equal(alpha[model.item_counts > 0], 0.50)


def test_api_uses_fold_in_immediately_after_five_linked_ratings(tmp_path, monkeypatch):
    import joblib
    import backend
    from fastapi.testclient import TestClient

    model = fixture_model()
    artifact = tmp_path / 'model.joblib'
    joblib.dump(model, artifact)
    monkeypatch.setenv('APP_ENV', 'test')
    monkeypatch.setenv('MODEL_PATH', str(artifact))
    monkeypatch.setenv('APP_DB_PATH', str(tmp_path / 'foldin.sqlite3'))
    backend._model = None
    backend._model_mtime = None
    backend.init_db()
    with TestClient(backend.app) as client:
        account = client.post('/users', json={'username': 'foldin_test_user'}).json()
        headers = {'Authorization': f"Bearer {account['token']}"}
        for mid in range(100, 104):
            result = client.put('/ratings', headers=headers, json={'tmdb_id': mid, 'rating': 4.5})
            assert result.status_code == 200
        cold = client.get('/recommendations?top_n=8', headers=headers).json()
        assert cold['recommendation_strategy'] == 'topk_popularity'
        assert cold['scoring']['fold_in_enabled'] is True
        result = client.put('/ratings', headers=headers, json={'tmdb_id': 104, 'rating': 5.0})
        assert result.status_code == 200
        folded = client.get('/recommendations?top_n=8', headers=headers).json()
        assert folded['recommendation_strategy'] == 'fold_in_topk_hybrid'
        assert any(item['collaborative_weight'] == 0.25 for item in folded['items'])
        assert all(item['tmdb_id'] not in range(100, 105) for item in folded['items'])
        assert folded['scoring']['fold_in_min_ratings'] == 5
        assert backend.get_model().user_to_row.get(backend.APP_USER_OFFSET + account['user_id']) is None
        assert client.delete('/ratings/104', headers=headers).status_code == 204
        fallback = client.get('/recommendations', headers=headers).json()
        assert fallback['recommendation_strategy'] == 'topk_popularity'
