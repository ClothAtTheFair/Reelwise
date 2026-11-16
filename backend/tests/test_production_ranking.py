import numpy as np
import pandas as pd
from recommender import HybridRecommender


def test_production_collaborative_weight_and_unrated_movie():
    movies = pd.DataFrame([
        dict(id=10, movieId=1, title='Space', genres='Science Fiction', overview='Futuristic space travel'),
        dict(id=20, movieId=2, title='Action', genres='Action', overview='Fantastic action thriller'),
        dict(id=30, movieId=None, title='Unrated', genres='Science Fiction', overview='Future planetary voyage'),
    ])
    ratings = pd.DataFrame([
        dict(userId=1, movieId=1, rating=5.0),
        dict(userId=1, movieId=2, rating=4.0),
        dict(userId=2, movieId=1, rating=3.0),
        dict(userId=2, movieId=2, rating=5.0),
    ])
    model = HybridRecommender(factors=1).fit(movies, ratings)
    scores, hist, alpha, strategy = model._ranking_scores(user_id=1)
    assert strategy == 'collaborative_topk_hybrid'
    np.testing.assert_allclose(alpha, [0.9, 0.9, 0.0])
    components, _, _ = model.score_components(user_id=1)
    content = model.score_topk_content(hist, top_k=10, popularity_weight=0.5)
    np.testing.assert_allclose(scores[:2], .9 * components['collaborative'][:2] + .1 * content[:2])
    np.testing.assert_allclose(scores[2], content[2])
    no_history, _, alpha2, strategy2 = model._ranking_scores()
    assert strategy2 == 'popularity'
    assert np.all(alpha2 == 0)
    np.testing.assert_allclose(no_history, model._ranking_scores(additional_ratings={10: 2.0})[0])
