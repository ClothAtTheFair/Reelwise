import numpy as np
import pandas as pd

from evaluate import MODES, score_models, make_charts
from recommender import HybridRecommender
from test_pipeline import sample
from filter_catalog import build_catalog


def test_modes_and_evaluation(tmp_path):
    tmdb, ratings, links = sample()
    movies, selected_ratings = build_catalog(tmdb, ratings, links, size=6, min_ratings=1)
    model = HybridRecommender(factors=2).fit(movies, selected_ratings)
    scores, history, alpha = model.score_components(1)
    assert set(scores) == set(MODES)
    assert all(len(values) == len(movies) for values in scores.values())
    for mode in MODES:
        prediction = model.predict(1, 1, movie_id_type='movielens', mode=mode)
        assert np.isfinite(prediction)
    comparison, curves, activity, meta = score_models(
        movies, selected_ratings, factors=2, max_rating_users=4,
        ranking_users=4, candidate_negatives=2, min_user_ratings=3)
    assert set(comparison.model) == set(MODES)
    assert (comparison.evaluated_ratings == 4).all()
    assert len(curves) == len(MODES) * 4
    assert meta['ranking_mode'] == 'sampled'
    make_charts(comparison, curves, activity, tmp_path)
    assert (tmp_path / 'rmse_comparison.png').exists()


def test_full_catalog_mode():
    tmdb, ratings, links = sample()
    movies, selected_ratings = build_catalog(tmdb, ratings, links, size=6, min_ratings=1)
    comparison, _, _, meta = score_models(
        movies, selected_ratings, factors=2, max_rating_users=4,
        ranking_users=4, ranking_mode='full', min_user_ratings=3)
    assert meta['ranking_mode'] == 'full'
    assert (comparison.ranking_users == meta['ranking_users']).all()


def test_sweep_cold_start_and_determinism():
    from evaluate import fixed_weight_scores
    tmdb, ratings, links = sample()
    movies, selected_ratings = build_catalog(tmdb, ratings, links, size=6, min_ratings=1)
    args = dict(factors=2, max_rating_users=4, ranking_users=4,
                ranking_mode='full', min_user_ratings=3,
                sweep_weights=(0, 0.5, 1), return_extended=True, seed=17)
    first = score_models(movies, selected_ratings, **args)
    second = score_models(movies, selected_ratings, **args)
    _, _, _, _, sweep, cold = first
    pd.testing.assert_frame_equal(sweep, second[4])
    assert list(sweep.collaborative_weight) == [0.0, 0.5, 1.0]
    assert set(cold.dimension) == {'user', 'movie'}
    assert set(cold.model) >= set(MODES)
    assert (sweep.ranking_users >= 0).all()
    # Unknown-item fallback always uses content, regardless of fixed blend.
    comp = {'collaborative': np.array([4.0, 4.0]),
            'content': np.array([2.0, 2.5])}
    assert fixed_weight_scores(comp, 1.0, True, np.array([0, 2])).tolist() == [2.0, 4.0]
