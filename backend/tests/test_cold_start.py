import numpy as np
import pandas as pd

from recommender import HybridRecommender
from cold_start_experiments import user_experiment, evaluate_user_targets


def dataset():
    movies = pd.DataFrame({
        'id': list(range(100, 118)),
        'movieId': list(range(1, 19)),
        'title': [f'Movie {i}' for i in range(18)],
        'genres': ['Action' if i % 2 else 'Drama' for i in range(18)],
        'overview': [f'This is an entertaining story number {i} of heroes and quests' for i in range(18)],
        'vote_count': [100 + i for i in range(18)],
        'vote_average': [7.0] * 18,
    })
    ratings = pd.DataFrame([
        {'userId': uid, 'movieId': mid, 'rating': 5.0 if mid % 3 else 2.0, 'timestamp': mid}
        for uid in range(1, 5) for mid in range(1, 16)
    ])
    return movies, ratings


def test_positive_only_weights():
    movies, ratings = dataset()
    model = HybridRecommender(factors=2).fit(movies, ratings)
    # The profile includes only positive evidence above three stars.
    h = {100: 5.0, 101: 4.0, 102: 2.0}
    p = model._profile(h)
    indices = [model.id_to_col[i] for i in h]
    weights = np.array(list(h.values()), dtype=np.float32)
    weights = np.maximum(weights - 3.0, 0.0)
    manual = np.asarray(model.features[indices].multiply(weights[:, None]).sum(axis=0)).reshape(1, -1)
    manual /= np.linalg.norm(manual)
    np.testing.assert_allclose(p, manual)
    assert model._profile({100: 5.0}) is not None
    assert model._profile({100: 2.0}) is None


def test_multi_target_evaluation_and_determinism():
    movies, ratings = dataset()
    a, summary, rankings = user_experiment(
        movies, ratings, users=2, histories=(0, 1, 3),
        holdout_positives=3, factors=2, negatives=8, seed=42,
        return_rankings=True,
    )
    assert len(a) == 2 * 3 * 3 * 4
    assert len(rankings) == 2 * 3 * 4
    assert set(summary.users) == {2}
    assert set(summary.evaluated_ratings) == {6}
    assert (rankings.hits_at_10 <= 3).all()
    assert np.allclose(rankings.recall_at_10, rankings.hits_at_10 / 3)
    assert np.allclose(rankings.precision_at_10, rankings.hits_at_10 / 10)
    assert rankings.ndcg_at_10.between(0, 1).all()
    b, bs, br = user_experiment(
        movies, ratings, users=2, histories=(0, 1, 3),
        holdout_positives=3, factors=2, negatives=8, seed=42,
        return_rankings=True,
    )
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(summary, bs)
    pd.testing.assert_frame_equal(rankings, br)


def test_topk_similarity_popularity_and_user_experiment():
    movies, ratings = dataset()
    model = HybridRecommender(factors=2).fit(movies, ratings)
    history = {100: 5.0, 101: 4.5, 102: 2.0}
    plain = model.score_topk_content(history, top_k=1, popularity_weight=0.0, chunk_size=5)
    # A movie's cosine similarity to itself is 1.0.
    assert np.isclose(plain[model.id_to_col[100]], 5.0)
    assert plain.min() >= 0.5 and plain.max() <= 5.0
    with_prior = model.score_topk_content(history, top_k=1, popularity_weight=0.15, chunk_size=5)
    assert with_prior.shape == plain.shape
    assert np.isfinite(with_prior).all()
    # No liked movies: matching fallback for the no-history condition.
    fallback = model.score_topk_content({})
    components, _, _ = model.score_components()
    np.testing.assert_allclose(fallback, components['content'])
    raw, summary, rankings = user_experiment(
        movies, ratings, users=2, histories=(0, 1, 3),
        holdout_positives=3, factors=2, negatives=8, seed=42,
        return_rankings=True, compare_topk=True, top_k=2, popularity_weight=0.15,
    )
    assert {'topk', 'topk_pop'} <= set(summary.model)
    assert len(raw) == 2 * 3 * 3 * 6
    assert rankings.ndcg_at_10.between(0, 1).all()


def test_popularity_sweep_same_candidates_and_pure_popularity():
    from cold_start_experiments import (
        popularity_sweep_scores, popularity_prior, sweep_tables,
        validate_sweep_weights,
    )
    import pytest
    movies, ratings = dataset()
    model = HybridRecommender(factors=2).fit(movies, ratings)
    weights = (0.0, 0.15, 1.0)
    h = {100: 5.0, 101: 4.5}
    scores = popularity_sweep_scores(model, h, weights, top_k=2)
    np.testing.assert_allclose(
        scores['topk_pop_w0.0000'],
        model.score_topk_content(h, top_k=2, popularity_weight=0.0)
    )
    np.testing.assert_allclose(
        scores['topk_pop_w1.0000'],
        0.5 + 4.5 * popularity_prior(model), rtol=1e-6
    )
    np.testing.assert_allclose(
        scores['topk_pop_w0.1500'],
        0.85 * scores['topk_pop_w0.0000'] + 0.15 * scores['topk_pop_w1.0000'],
        atol=1e-5
    )
    with pytest.raises(ValueError):
        validate_sweep_weights((0, 1.1))
    with pytest.raises(ValueError):
        validate_sweep_weights((0.1, 0.1))
    raw, _, ranking = user_experiment(
        movies, ratings, users=2, histories=(0, 1, 3),
        holdout_positives=3, factors=2, negatives=8, seed=42,
        return_rankings=True, sweep_weights=weights, top_k=2
    )
    per_user, summary = sweep_tables(ranking, raw, weights)
    assert set(summary.popularity_weight) == set(weights)
    assert (summary.users == 2).all()
    assert set(summary.revealed_ratings) == {0, 1, 3}
    assert len(per_user) == 2 * 3 * len(weights)
    assert (summary.ndcg_at_10.between(0, 1)).all()
    # Pure popularity ranks the same candidates at every history length.
    pop = per_user[per_user.popularity_weight == 1.0]
    assert pop.groupby('user_id').ndcg_at_10.nunique().eq(1).all()
