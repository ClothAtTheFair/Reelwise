import numpy as np
import pandas as pd
from filter_catalog import build_catalog
from recommender import HybridRecommender
from train import temporal_split, evaluate


def sample():
    movies = pd.DataFrame({
        'id': [10, 20, 30, 40, 50, 60],
        'title': ['Space voyage', 'Moon trip', 'Romantic drama', 'Action fighter', 'Ocean trip', 'Quiet life'],
        'overview': ['An astronaut explores the vast solar system beyond planet earth.'] * 6,
        'genres': ['Science Fiction', 'Science Fiction', 'Romance', 'Action', 'Adventure', 'Drama'],
        'keywords': ['space', 'moon', 'romance', 'combat', 'sea', 'peace'],
        'tagline': ['test'] * 6,
        'status': ['Released'] * 6,
        'adult': [False] * 6,
        'vote_count': [500, 400, 300, 200, 100, 50],
        'popularity': [6, 5, 4, 3, 2, 1],
    })
    links = pd.DataFrame({'movieId': [1, 2, 3, 4], 'tmdbId': [10, 20, 30, 40]})
    ratings = pd.DataFrame([
        {'userId': user, 'movieId': mid, 'rating': 4.5 if mid < 3 else 2.0, 'timestamp': mid + 10 * user}
        for user in range(1, 5) for mid in range(1, 5)
    ])
    return movies, ratings, links


def test_selection_and_predictions():
    tmdb, ratings, links = sample()
    selected, chosen = build_catalog(tmdb, ratings, links, size=6, min_ratings=2)
    assert len(selected) == 6
    assert len(chosen) == 16
    assert selected.id.nunique() == 6
    model = HybridRecommender(factors=2).fit(selected, chosen)
    recs = model.recommend(user_id=1, top_n=5)
    assert {r['tmdb_id'] for r in recs} == {50, 60}
    assert all(r['collaborative_weight'] == 0 for r in recs)
    assert 0.5 <= model.predict(1, 1, movie_id_type='movielens') <= 5
    cold = model.recommend(additional_ratings={10: 5}, top_n=3)
    assert len(cold) == 3 and all(r['tmdb_id'] != 10 for r in cold)
    assert all(r['collaborative_weight'] == 0 for r in cold)


def test_split_and_evaluate():
    tmdb, ratings, links = sample()
    selected, chosen = build_catalog(tmdb, ratings, links, size=6, min_ratings=1)
    train, test = temporal_split(chosen, min_user_ratings=3)
    assert len(test) == 4
    assert len(train) == 12
    model = HybridRecommender(factors=2).fit(selected, train)
    metrics = evaluate(model, test, max_ranking_users=3)
    assert metrics['evaluated_ratings'] == 4
    assert np.isfinite(metrics['rmse'])
