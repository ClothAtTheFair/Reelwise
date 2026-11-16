import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from recommender import HybridRecommender


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import backend
    monkeypatch.setenv('APP_ENV', 'test')
    movies = pd.DataFrame([
        {'id': 11, 'movieId': 1, 'title': 'Galaxy Film', 'genres': 'Sci-Fi', 'overview': 'Galaxy journey space future robots', 'vote_count': 100, 'vote_average': 8.0},
        {'id': 22, 'movieId': 2, 'title': 'Comedy Film', 'genres': 'Comedy', 'overview': 'Funny silly family jokes and laughs', 'vote_count': 70, 'vote_average': 7.0},
        {'id': 33, 'movieId': 3, 'title': 'Galaxy Return', 'genres': 'Sci-Fi', 'overview': 'Space galaxy universe future journey', 'vote_count': 40, 'vote_average': 7.5},
        {'id': 44, 'movieId': None, 'title': 'New Galaxy', 'genres': 'Sci-Fi', 'overview': 'A space trip with future starships', 'vote_count': 15, 'vote_average': 7.0},
    ])
    ratings = pd.DataFrame([
        {'userId': 1, 'movieId': 1, 'rating': 5},
        {'userId': 1, 'movieId': 2, 'rating': 2},
        {'userId': 2, 'movieId': 1, 'rating': 4},
        {'userId': 2, 'movieId': 3, 'rating': 5},
    ])
    model = HybridRecommender(factors=1).fit(movies, ratings)
    model_path = tmp_path / 'model.joblib'
    joblib.dump(model, model_path)
    monkeypatch.setenv('MODEL_PATH', str(model_path))
    monkeypatch.setenv('APP_DB_PATH', str(tmp_path / 'users.sqlite3'))
    backend._model = None
    backend._model_mtime = None
    backend.init_db()
    with TestClient(backend.app) as c:
        yield c


def test_user_rating_recommendations_and_auth(client):
    assert client.get('/health').json()['model_ready']
    assert client.get('/recommendations').status_code == 401
    created = client.post('/users', json={'username': 'student1'})
    assert created.status_code == 201
    token = created.json()['token']
    headers = {'Authorization': f'Bearer {token}'}
    assert client.post('/users', json={'username': 'student1'}).status_code == 409
    assert client.get('/movies?q=galaxy').json()['total'] == 3
    assert client.get('/movies/44').status_code == 200
    assert client.get('/movies/999').status_code == 404
    assert client.put('/ratings', headers=headers, json={'tmdb_id': 11, 'rating': 4.3}).status_code == 422
    assert client.put('/ratings', headers=headers, json={'tmdb_id': 999, 'rating': 5}).status_code == 404
    assert client.put('/ratings', headers=headers, json={'tmdb_id': 11, 'rating': 5}).status_code == 200
    assert client.put('/ratings', headers=headers, json={'tmdb_id': 22, 'rating': 1}).status_code == 200
    assert len(client.get('/ratings', headers=headers).json()) == 2
    result = client.get('/recommendations?top_n=3', headers=headers)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data['personalization'] == 'topk_popularity'
    assert data['scoring']['top_k'] == 10
    assert data['scoring']['popularity_weight'] == 0.5
    assert data['scoring']['score_type'] == 'ranking_score_not_predicted_stars'
    assert all(m['tmdb_id'] not in (11, 22) for m in data['items'])
    assert any(m['tmdb_id'] == 44 for m in data['items'])
    assert client.delete('/ratings/11', headers=headers).status_code == 204
    assert len(client.get('/ratings', headers=headers).json()) == 1


def test_app_ratings_export_for_training(client, tmp_path, monkeypatch):
    import os
    from train import read_app_ratings
    registered = client.post('/users', json={'username': 'student2'}).json()
    headers = {'Authorization': f"Bearer {registered['token']}"}
    client.put('/ratings', headers=headers, json={'tmdb_id': 11, 'rating': 4.5})
    client.put('/ratings', headers=headers, json={'tmdb_id': 44, 'rating': 5})
    model = __import__('backend').get_model()
    df = read_app_ratings(__import__('pathlib').Path(os.environ['APP_DB_PATH']), model.movies)
    assert len(df) == 1  # Movie 44 has no MovieLens ID
    assert int(df.iloc[0]['userId']) == 1_000_000_000 + registered['user_id']


def test_genres_persist_are_private_and_affect_recommendations(client):
    first = client.post('/users', json={'username': 'genres1'}).json()
    second = client.post('/users', json={'username': 'genres2'}).json()
    h1 = {'Authorization': f"Bearer {first['token']}"}
    h2 = {'Authorization': f"Bearer {second['token']}"}
    assert client.get('/users/me/genres').status_code == 401
    assert client.get('/users/me/genres', headers=h1).json() == {'genres': []}
    before = client.get('/recommendations?top_n=4', headers=h1).json()['items']
    before_scores = {m['tmdb_id']: m['score'] for m in before}
    assert client.get('/genres').json()['genres']
    updated = client.put('/users/me/genres', headers=h1, json={'genres': ['comedy', 'COMEDY']})
    assert updated.status_code == 200
    assert updated.json() == {'genres': ['Comedy']}
    assert client.get('/users/me/genres', headers=h2).json() == {'genres': []}
    result = client.get('/recommendations?top_n=4', headers=h1).json()
    assert result['preferred_genres'] == ['Comedy']
    after_scores = {m['tmdb_id']: m for m in result['items']}
    assert after_scores[22]['genre_match'] == 1
    assert after_scores[22]['score'] > before_scores[22]
    assert client.put('/users/me/genres', headers=h1, json={'genres': ['Unknown Genre']}).status_code == 422
    assert client.get('/users/me/genres', headers=h1).json() == {'genres': ['Comedy']}
    assert client.put('/users/me/genres', headers=h1, json={'genres': []}).status_code == 200
    assert client.get('/users/me/genres', headers=h1).json() == {'genres': []}


def test_frontend_ranking_modes_and_collaborative_fallback(client, monkeypatch):
    import backend
    user = client.post('/users', json={'username': 'trained_user'}).json()
    h = {'Authorization': f"Bearer {user['token']}"}
    zero = client.get('/recommendations?top_n=4', headers=h).json()
    assert zero['recommendation_strategy'] == 'popularity'
    assert all(m['collaborative_weight'] == 0 for m in zero['items'])
    client.put('/ratings', headers=h, json={'tmdb_id': 11, 'rating': 5})
    cold = client.get('/recommendations?top_n=4', headers=h).json()
    assert cold['recommendation_strategy'] == 'topk_popularity'
    assert all(m['tmdb_id'] != 11 for m in cold['items'])
    model = backend.get_model()
    # Mimic a periodic retrain that learned this application's current ratings.
    internal = backend.APP_USER_OFFSET + user['user_id']
    model.user_ratings[internal] = {11: 5.0}
    model.user_to_row[internal] = model.user_to_row[1]
    model.user_means[internal] = 5.0
    model.user_counts[internal] = 1
    trained = client.get('/recommendations?top_n=4', headers=h).json()
    assert trained['recommendation_strategy'] == 'collaborative_topk_hybrid'
    by_id = {m['tmdb_id']: m for m in trained['items']}
    assert by_id[22]['collaborative_weight'] == 0.9
    assert by_id[44]['collaborative_weight'] == 0.0
    # A newly saved rating makes the previously fitted user representation stale.
    client.put('/ratings', headers=h, json={'tmdb_id': 22, 'rating': 4})
    changed = client.get('/recommendations', headers=h).json()
    assert changed['recommendation_strategy'] == 'topk_popularity'


def test_app_snapshot_works_with_tmdb_only_ratings(client):
    import backend
    created = client.post('/users', json={'username': 'snapshot_user'}).json()
    h = {'Authorization': f"Bearer {created['token']}"}
    client.put('/ratings', headers=h, json={'tmdb_id': 11, 'rating': 5})
    client.put('/ratings', headers=h, json={'tmdb_id': 44, 'rating': 5})
    model = backend.get_model()
    uid = backend.APP_USER_OFFSET + created['user_id']
    model.user_ratings[uid] = {11: 5.0}
    model.user_to_row[uid] = model.user_to_row[1]
    model.user_means[uid] = 5.0
    model.user_counts[uid] = 1
    model.app_rating_snapshots = {uid: {11: 5.0, 44: 5.0}}
    assert client.get('/recommendations', headers=h).json()['recommendation_strategy'] == 'collaborative_topk_hybrid'
    client.delete('/ratings/44', headers=h)
    assert client.get('/recommendations', headers=h).json()['recommendation_strategy'] == 'topk_popularity'
