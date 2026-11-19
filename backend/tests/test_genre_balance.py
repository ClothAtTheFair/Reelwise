import numpy as np
import pandas as pd
import pytest
from recommender import HybridRecommender


def model_with_ranked_movies(monkeypatch, *, all_preferred=False):
    model = HybridRecommender()
    n = 24
    model.movies = pd.DataFrame({
        'id': np.arange(100, 100 + n),
        'title': [f'Movie {i}' for i in range(n)],
        'genres': ['Comedy' if i < 18 or all_preferred else 'Drama' for i in range(n)],
    })
    model.genre = model.movies.genres
    model.id_to_col = dict(zip(model.movies.id, range(n)))
    scores = np.linspace(4.8, 3.4, n, dtype=np.float32)
    def fake_ranking_scores(*args, **kwargs):
        return scores.copy(), {}, np.zeros(n, dtype=np.float32), 'topk_popularity'
    monkeypatch.setattr(model, '_ranking_scores', fake_ranking_scores)
    return model


def test_genre_prior_decreases_as_user_rates_movies():
    model = HybridRecommender()
    assert model.genre_preference_weight(0) == pytest.approx(0.08)
    assert model.genre_preference_weight(5) == pytest.approx(0.04)
    assert model.genre_preference_weight(20) == pytest.approx(0.016)
    assert model.genre_preference_weight(100) < model.genre_preference_weight(20)


def test_preferred_genre_no_longer_monopolizes_top_ten(monkeypatch):
    model = model_with_ranked_movies(monkeypatch)
    original = model.recommend(top_n=10)
    assert all('Comedy' in m['genres'] for m in original)
    balanced = model.recommend(top_n=10, preferred_genres=['Comedy'])
    assert len(balanced) == 10
    assert len(set(m['tmdb_id'] for m in balanced)) == 10
    assert sum('Comedy' in m['genres'] for m in balanced) == 6
    assert sum('Drama' in m['genres'] for m in balanced) == 4
    assert all(np.isfinite(m['score']) for m in balanced)


def test_diversity_guard_backfills_when_no_alternatives(monkeypatch):
    model = model_with_ranked_movies(monkeypatch, all_preferred=True)
    balanced = model.recommend(top_n=10, preferred_genres=['Comedy'])
    assert len(balanced) == 10
    assert all('Comedy' in m['genres'] for m in balanced)


def test_selected_genre_still_boosts_tied_movies(monkeypatch):
    model = model_with_ranked_movies(monkeypatch)
    initial = np.linspace(4.8, 3.4, len(model.movies), dtype=np.float32)
    adjusted, matches = model._add_genre_preferences(initial, ['Comedy'], model.genre_preference_weight(20))
    assert adjusted[0] > initial[0]
    assert matches[0] == 1.0
    assert adjusted[-1] < initial[-1]
