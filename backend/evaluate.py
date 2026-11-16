"""Compare recommendation approaches on identical, temporal holdout examples.

Examples:
    python evaluate.py --max-rating-users 2000 --ranking-users 100
    python evaluate.py --ranking-mode full --ranking-users 100

Caveats: the processed movie catalog may have been selected using all ratings;
for a strict leakage-free benchmark, select the catalog using training data only.
One held-out relevant movie/user caps Precision@10 at 0.1.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from recommender import HybridRecommender
from train import temporal_split

MODES = ('baseline', 'collaborative', 'content', 'hybrid')
K_VALUES = (5, 10, 20, 50)
DEFAULT_SWEEP = (0.0, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0)


def activity_bucket(n):
    return "1-9" if n < 10 else "10-49" if n < 50 else "50-199" if n < 200 else "200+"


def item_bucket(n):
    return "0" if n == 0 else "1-4" if n < 5 else "5-19" if n < 20 else "20+"


def metrics_from_values(actual, predicted, ranks):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    ranks = np.asarray(ranks, dtype=int)
    mae = float(np.mean(np.abs(actual - predicted))) if len(actual) else np.nan
    rmse = float(np.sqrt(np.mean((actual - predicted)**2))) if len(actual) else np.nan
    hits = ranks <= 10
    return {"evaluated_ratings": len(actual), "ranking_users": len(ranks),
            "mae": mae, "rmse": rmse,
            "precision_at_10": float(hits.mean() / 10) if len(ranks) else np.nan,
            "recall_at_10": float(hits.mean()) if len(ranks) else np.nan,
            "ndcg_at_10": float(np.mean(np.where(hits, 1 / np.log2(ranks + 1), 0))) if len(ranks) else np.nan}


def fixed_weight_scores(components, weight, has_user_factors, item_counts):
    """Collaborative share on known user/items; content fallback for unknowns."""
    confidence = np.asarray(item_counts > 0, dtype=float) if has_user_factors else 0.0
    a = weight * confidence
    return a * components["collaborative"] + (1 - a) * components["content"]


def score_models(movies, ratings, *, factors=40, max_rating_users=2000,
                 ranking_users=100, candidate_negatives=200, ranking_mode='sampled',
                 min_user_ratings=5, seed=42, sweep_weights=None, return_extended=False):
    """Fit once and evaluate four modes on the same holdout and candidate sets."""
    if ranking_mode not in ('sampled', 'full'):
        raise ValueError('ranking_mode must be sampled or full')
    if sweep_weights is None:
        sweep_weights = DEFAULT_SWEEP
    sweep_weights = tuple(sorted(set(float(w) for w in sweep_weights)))
    if any(not np.isfinite(w) or not 0 <= w <= 1 for w in sweep_weights):
        raise ValueError('Sweep weights must be between 0 and 1')
    train, test = temporal_split(ratings, min_user_ratings=min_user_ratings)
    if test.empty:
        raise ValueError('No eligible users for holdout; lower --min-user-ratings')
    # Remove any precomputed rating counts that might reflect held-out data.
    prepared_movies = movies.copy()
    counts = train.groupby('movieId').size()
    prepared_movies['rating_count'] = prepared_movies['movieId'].map(counts).fillna(0).astype(int)
    model = HybridRecommender(factors=factors, random_state=seed).fit(prepared_movies, train)
    test = test[test.movieId.isin(model.movie_to_tmdb)].copy()
    if test.empty:
        raise ValueError('No held-out movies were linked to catalog')
    rng = np.random.default_rng(seed)
    # Sample users rather than interactions; exactly one test row/user here.
    rating_test = test.sample(n=min(max_rating_users, len(test)), random_state=seed) if max_rating_users > 0 else test
    positive_test = test[test.rating >= 4].sample(frac=1, random_state=seed).head(ranking_users)
    ranking_map = {int(row.userId): row for row in positive_test.itertuples(index=False)}
    rating_map = {int(row.userId): row for row in rating_test.itertuples(index=False)}
    all_users = sorted(set(rating_map) | set(ranking_map))
    variants = {f'fixed_{w:.3f}': w for w in sweep_weights}
    names = list(MODES) + list(variants)
    metrics = {mode: {'actual': [], 'predicted': [], 'rank': [], 'activity': [],
                      'rating_items': [], 'rank_activity': [], 'rank_items': []} for mode in names}
    candidate_ids = model.movies.id.to_numpy(dtype=np.int64)
    max_k = max(K_VALUES)
    ranking_n = 0
    for uid in all_users:
        components, history, _ = model.score_components(user_id=uid)
        has_user_factors = model.svd is not None and uid in model.user_to_row
        scores = dict(components)
        for name, weight in variants.items():
            scores[name] = fixed_weight_scores(components, weight, has_user_factors, model.item_counts)
        if uid in rating_map:
            row = rating_map[uid]
            col = model.id_to_col[model.movie_to_tmdb[int(row.movieId)]]
            for mode in names:
                metrics[mode]['actual'].append(float(row.rating))
                metrics[mode]['predicted'].append(float(scores[mode][col]))
                metrics[mode]['activity'].append(len(history))
                metrics[mode]['rating_items'].append(int(model.item_counts[col]))
        if uid not in ranking_map:
            continue
        row = ranking_map[uid]
        target = model.movie_to_tmdb.get(int(row.movieId))
        if target is None:
            continue
        # Exclude all movies with known training ratings; no test positives in negatives.
        forbidden = set(history) | {target}
        eligible = candidate_ids[~np.isin(candidate_ids, list(forbidden))]
        if ranking_mode == 'sampled':
            negatives = rng.choice(eligible, size=min(candidate_negatives, len(eligible)), replace=False)
            selected = np.append(negatives, target)
        else:
            selected = np.append(eligible, target)
        cols = np.array([model.id_to_col[int(mid)] for mid in selected], dtype=np.int64)
        target_col = model.id_to_col[int(target)]
        target_item_n = int(model.item_counts[target_col])
        for mode in names:
            values = scores[mode][cols]
            # Rank in O(candidates) rather than sorting the whole 150k catalog.
            # Exact tie-breaking matches lexsort((selected, -values)).
            target_score = scores[mode][target_col]
            rank = 1 + int(np.count_nonzero(
                (values > target_score) |
                ((values == target_score) & (selected < target))
            ))
            metrics[mode]['rank'].append(rank)
            metrics[mode]['rank_activity'].append(len(history))
            metrics[mode]['rank_items'].append(target_item_n)
        ranking_n += 1

    comparison, curves, activity_rows = [], [], []
    for mode in MODES:
        m = metrics[mode]
        actual = np.asarray(m['actual'])
        predicted = np.asarray(m['predicted'])
        ranks = np.asarray(m['rank'], dtype=int)
        mae = float(np.abs(actual - predicted).mean()) if len(actual) else np.nan
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2))) if len(actual) else np.nan
        def ranking_at(k):
            if not len(ranks):
                return np.nan, np.nan, np.nan
            found = ranks <= k
            return (float(found.mean() / k), float(found.mean()),
                    float(np.mean(np.where(found, 1 / np.log2(ranks + 1), 0))))
        p10, r10, n10 = ranking_at(10)
        comparison.append({'model': mode, 'evaluated_ratings': len(actual),
                           'ranking_users': len(ranks), 'mae': mae, 'rmse': rmse,
                           'precision_at_10': p10, 'recall_at_10': r10, 'ndcg_at_10': n10})
        for k in K_VALUES:
            p, r, n = ranking_at(k)
            curves.append({'model': mode, 'k': k, 'precision': p, 'recall': r, 'ndcg': n})
        activity = np.asarray(m['activity'])
        for label, mask in [('1-9', activity < 10), ('10-49', (activity >= 10) & (activity < 50)),
                            ('50-199', (activity >= 50) & (activity < 200)), ('200+', activity >= 200)]:
            activity_rows.append({'model': mode, 'user_rating_bucket': label, 'n': int(mask.sum()),
                                  'rmse': float(np.sqrt(np.mean((actual[mask] - predicted[mask])**2)))
                                  if np.any(mask) else np.nan})
    # A sweep of fixed collaborative weights, plus per-user and per-item cold-start slices.
    sweep = []
    cold_start = []
    for name in names:
        m = metrics[name]
        if name in variants:
            sweep.append({'collaborative_weight': variants[name],
                          **metrics_from_values(m['actual'], m['predicted'], m['rank'])})
        for dimension, values_r, values_k in (
                ('user', m['activity'], m['rank_activity']),
                ('movie', m['rating_items'], m['rank_items'])):
            fn = activity_bucket if dimension == 'user' else item_bucket
            labels = sorted(set(map(fn, values_r)) | set(map(fn, values_k)))
            for label in labels:
                rating_idx = [i for i, x in enumerate(values_r) if fn(x) == label]
                ranking_idx = [i for i, x in enumerate(values_k) if fn(x) == label]
                row = metrics_from_values(
                    [m['actual'][i] for i in rating_idx],
                    [m['predicted'][i] for i in rating_idx],
                    [m['rank'][i] for i in ranking_idx])
                cold_start.append({'model': name, 'dimension': dimension,
                                   'bucket': label, **row})
    meta = {'train_ratings': len(train), 'test_ratings': len(test),
            'accuracy_users': len(rating_map), 'ranking_users': ranking_n,
            'ranking_mode': ranking_mode, 'candidate_negatives': candidate_negatives if ranking_mode == 'sampled' else None,
            'seed': seed, 'factors': factors,
            'ranking_definition': 'One positive held-out rating >=4 per user; unobserved movies sampled as assumed negatives',
            'limitation': 'Catalog may have been selected with test interactions; fix catalog selection for strict leakage-free benchmark.'}
    if return_extended:
        return (pd.DataFrame(comparison), pd.DataFrame(curves),
                pd.DataFrame(activity_rows), meta,
                pd.DataFrame(sweep), pd.DataFrame(cold_start))
    return pd.DataFrame(comparison), pd.DataFrame(curves), pd.DataFrame(activity_rows), meta


def make_charts(comparison, curves, activity, directory, sweep=None, cold_start=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save(name):
        plt.tight_layout()
        plt.savefig(directory / name, dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(comparison.model, comparison.rmse)
    plt.ylabel('RMSE (stars; lower is better)')
    plt.title('Holdout rating prediction accuracy')
    save('rmse_comparison.png')

    plt.figure(figsize=(8, 4.5))
    for mode, grp in curves.groupby('model', sort=False):
        plt.plot(grp.k, grp.ndcg, marker='o', label=mode)
    plt.xlabel('K')
    plt.ylabel('NDCG@K (higher is better)')
    plt.title('Held-out ranking comparison')
    plt.legend()
    save('ndcg_at_k.png')

    pivot = activity.pivot(index='user_rating_bucket', columns='model', values='rmse')
    pivot = pivot.reindex(['1-9', '10-49', '50-199', '200+'])
    plt.figure(figsize=(8, 4.5))
    for mode in MODES:
        plt.plot(pivot.index, pivot[mode], marker='o', label=mode)
    plt.xlabel('Training ratings per user')
    plt.ylabel('RMSE (stars)')
    plt.title('Rating error by user history')
    plt.legend()
    save('performance_by_user_activity.png')

    if sweep is not None and not sweep.empty:
        plt.figure(figsize=(8, 4.5))
        plt.plot(sweep.collaborative_weight, sweep.rmse, marker='o', label='RMSE')
        plt.xlabel('Fixed collaborative weight')
        plt.ylabel('RMSE (stars; lower is better)')
        plt.title('Hybrid weight sweep: rating accuracy')
        save('hybrid_weight_rmse.png')

        plt.figure(figsize=(8, 4.5))
        plt.plot(sweep.collaborative_weight, sweep.ndcg_at_10, marker='o')
        plt.xlabel('Fixed collaborative weight')
        plt.ylabel('NDCG@10 (higher is better)')
        plt.title('Hybrid weight sweep: ranking quality')
        save('hybrid_weight_ndcg.png')

    if cold_start is not None and not cold_start.empty:
        subset = cold_start[(cold_start.dimension == 'movie') &
                            (cold_start.model.isin(MODES))]
        pivot = subset.pivot(index='bucket', columns='model', values='rmse')
        pivot = pivot.reindex(['0', '1-4', '5-19', '20+'])
        plt.figure(figsize=(8, 4.5))
        for mode in MODES:
            plt.plot(pivot.index, pivot[mode], marker='o', label=mode)
        plt.xlabel('Training ratings for held-out movie')
        plt.ylabel('RMSE (stars; lower is better)')
        plt.title('Movie cold-start: rating error')
        plt.legend()
        save('cold_start_movie_rmse.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--factors', type=int, default=40)
    parser.add_argument('--max-rating-users', type=int, default=2000,
                        help='Subsample accuracy evaluation for reasonable runtime; 0 = all')
    parser.add_argument('--ranking-users', type=int, default=100)
    parser.add_argument('--candidates', type=int, default=200, help='Sampled negatives per user')
    parser.add_argument('--ranking-mode', choices=['sampled', 'full'], default='sampled')
    parser.add_argument('--min-user-ratings', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-charts', action='store_true')
    parser.add_argument('--sweep-weights', type=float, nargs='+', default=list(DEFAULT_SWEEP),
                        help='Fixed collaborative weights to compare; 0=content, 1=collab')
    args = parser.parse_args()
    if args.factors < 1 or args.max_rating_users < 0 or args.ranking_users < 0 or args.candidates < 1:
        parser.error('Invalid evaluation limits')
    movies = pd.read_csv(args.data / 'training_movies.csv', low_memory=False)
    ratings = pd.read_csv(args.data / 'training_ratings.csv')
    print(f'Fitting evaluation model on {len(ratings):,} ratings and {len(movies):,} catalog movies...')
    comp, curves, activity, meta, sweep, cold_start = score_models(
        movies, ratings, factors=args.factors, max_rating_users=args.max_rating_users,
        ranking_users=args.ranking_users, candidate_negatives=args.candidates,
        ranking_mode=args.ranking_mode, min_user_ratings=args.min_user_ratings, seed=args.seed,
        sweep_weights=args.sweep_weights, return_extended=True)
    args.output.mkdir(parents=True, exist_ok=True)
    comp.to_csv(args.output / 'model_comparison.csv', index=False)
    curves.to_csv(args.output / 'ndcg_at_k.csv', index=False)
    activity.to_csv(args.output / 'performance_by_user_activity.csv', index=False)
    sweep.to_csv(args.output / 'hybrid_weight_sweep.csv', index=False)
    cold_start.to_csv(args.output / 'cold_start_comparison.csv', index=False)
    (args.output / 'evaluation_metadata.json').write_text(json.dumps(meta, indent=2))
    if not args.no_charts:
        make_charts(comp, curves, activity, args.output, sweep, cold_start)
    print(comp.round(4).to_string(index=False))
    print('\nFixed-weight hybrid sweep:')
    print(sweep.round(4).to_string(index=False))
    print('\nCold-start group counts (hybrid):')
    print(cold_start[cold_start.model == 'hybrid'][['dimension', 'bucket', 'evaluated_ratings', 'ranking_users']].to_string(index=False))
    print('Saved evaluation results in:', args.output)
    print('NOTE: sampled ranking evaluates one relevant held-out movie against assumed negatives, not full user relevance.')
    print('NOTE: catalog selection may use holdout ratings; rerun selection on training data for a fully leakage-free benchmark.')


if __name__ == '__main__':
    main()
