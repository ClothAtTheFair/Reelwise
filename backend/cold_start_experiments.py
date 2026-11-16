"""Controlled cold-start experiments for HybridRecommender.

User: leave selected test users entirely out of training; reserve multiple
positive interactions per user and reveal 0/1/3/5/10/20 earlier ratings. This implementation CANNOT
fold new users into SVD: collaborative scores remain unpersonalized.

Movie: hold selected test movies' interactions to 0/1/5/20 examples before fit.
For each cap the model is refit; evaluate the same unseen test interactions.

Catalog caveat: the processed catalog may have been preselected using all ratings.
For strict leakage-free research, construct the catalog before this procedure
from metadata and training interactions only.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from recommender import HybridRecommender

MODES = ('baseline', 'collaborative', 'content', 'hybrid90')
TOPK_MODES = ('topk', 'topk_pop')
DEFAULT_SWEEP_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 1.0)


def prepare(movies, ratings):
    valid = set(pd.to_numeric(movies['movieId'], errors='coerce').dropna().astype('int64'))
    r = ratings[ratings.movieId.isin(valid)].copy()
    if 'timestamp' not in r:
        raise ValueError('ratings must include a timestamp column')
    return r.sort_values(['userId', 'timestamp', 'movieId']).reset_index(drop=True)


def fixed_scores(components, model, user_id=None, weight=0.9):
    """Content fallback if user or item has no fitted collaborative factor."""
    has_user = model.svd is not None and user_id is not None and int(user_id) in model.user_to_row
    confidence = (model.item_counts > 0).astype(float) if has_user else np.zeros(len(model.movies))
    a = weight * confidence
    return a * components['collaborative'] + (1 - a) * components['content']


def evaluate_target(model, target_tmdb, history, components, *, user_id=None,
                    negatives=200, full=False, rng=None):
    """One positive test movie against a fixed set of unrated candidates."""
    rng = rng or np.random.default_rng(42)
    all_ids = model.movies.id.to_numpy(dtype=np.int64)
    exclusion = set(history) | {int(target_tmdb)}
    pool = all_ids[~np.isin(all_ids, list(exclusion))]
    if not full and len(pool) > negatives:
        pool = rng.choice(pool, size=negatives, replace=False)
    choices = np.append(pool, target_tmdb)
    cols = np.array([model.id_to_col[int(i)] for i in choices], dtype=np.int64)
    col = model.id_to_col[int(target_tmdb)]
    components = {**components, 'hybrid90': fixed_scores(components, model, user_id)}
    result = {}
    for mode in MODES:
        s = components[mode]
        t = s[col]
        rank = 1 + int(np.count_nonzero((s[cols] > t) | ((s[cols] == t) & (choices < target_tmdb))))
        result[f'{mode}_rank'] = rank
        result[f'{mode}_ndcg10'] = 1 / np.log2(rank+1) if rank <= 10 else 0.0
        result[f'{mode}_hit10'] = int(rank <= 10)
        result[f'{mode}_prediction'] = float(s[col])
    return result


def summarize(examples, group):
    rows = []
    for key, frame in examples.groupby(group, sort=True):
        for mode in MODES:
            error = frame[f'{mode}_prediction'].to_numpy() - frame['actual'].to_numpy()
            rows.append({group: key, 'model': mode, 'n': len(frame),
                         'mae': float(np.abs(error).mean()),
                         'rmse': float(np.sqrt((error**2).mean())),
                         'hit_at_10': frame[f'{mode}_hit10'].mean(),
                         'ndcg_at_10': frame[f'{mode}_ndcg10'].mean()})
    return pd.DataFrame(rows)


def evaluate_user_targets(model, target_ids, history, components, *, excluded_ids,
                          negatives=200, full=False, rng=None, extra_components=None):
    """Rank all held-out positives together against one matched candidate set.

    Precision/recall/NDCG are computed once per user and reveal condition,
    rather than averaging single-positive NDCGs. Candidate selection is kept
    constant across history caps for each user by passing an identically seeded
    RNG and the same excluded IDs.
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    targets = np.asarray(target_ids, dtype=np.int64)
    all_ids = model.movies.id.to_numpy(dtype=np.int64)
    pool = all_ids[~np.isin(all_ids, list(set(excluded_ids) | set(target_ids)))]
    if not full and len(pool) > negatives:
        pool = rng.choice(pool, size=negatives, replace=False)
    choices = np.concatenate((pool, targets))
    cols = np.array([model.id_to_col[int(i)] for i in choices], dtype=np.int64)
    target_cols = np.array([model.id_to_col[int(i)] for i in targets], dtype=np.int64)
    components = {**components, 'hybrid90': fixed_scores(components, model, None)}
    modes = MODES
    if extra_components:
        components.update(extra_components)
        modes = MODES + tuple(extra_components)
    rankings, prediction_rows = {}, []
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(targets))))
    for mode in modes:
        scores = components[mode]
        # Reproducible tie-break by movie ID (ascending).
        sorted_positions = np.lexsort((choices, -scores[cols]))
        sorted_ids = choices[sorted_positions]
        ranks_by_id = {int(mid): rank for rank, mid in enumerate(sorted_ids, start=1)}
        ranks = [ranks_by_id[int(mid)] for mid in targets]
        hits = sum(r <= 10 for r in ranks)
        dcg = sum(1.0 / np.log2(r + 1) for r in ranks if r <= 10)
        rankings[mode] = {
            'hit_at_10': int(hits > 0),
            'hits_at_10': hits,
            'precision_at_10': hits / 10,
            'recall_at_10': hits / len(targets),
            'ndcg_at_10': dcg / ideal if ideal else 0.0,
        }
        for mid, col, rank in zip(targets, target_cols, ranks):
            prediction_rows.append({
                'tmdb_id': int(mid), 'model': mode, 'rank': rank,
                'prediction': float(scores[col]),
            })
    return rankings, prediction_rows


def summarize_user_experiment(target_rows, ranking_rows):
    """Macro-average ranking metrics by user; evaluate prediction errors by target."""
    rating_df = pd.DataFrame(target_rows)
    ranking_df = pd.DataFrame(ranking_rows)
    rows = []
    for (cap, mode), rank_group in ranking_df.groupby(['revealed_ratings', 'model']):
        predictions = rating_df[(rating_df.revealed_ratings == cap) &
                                (rating_df.model == mode)]
        error = predictions.prediction.to_numpy() - predictions.actual.to_numpy()
        rows.append({
            'revealed_ratings': int(cap), 'model': mode,
            'users': rank_group.user_id.nunique(),
            'evaluated_ratings': len(predictions),
            'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt(np.mean(error ** 2))),
            'hit_at_10': float(rank_group.hit_at_10.mean()),
            'precision_at_10': float(rank_group.precision_at_10.mean()),
            'recall_at_10': float(rank_group.recall_at_10.mean()),
            'ndcg_at_10': float(rank_group.ndcg_at_10.mean()),
            'median_target_rank': float(predictions['rank'].median()),
        })
    return pd.DataFrame(rows).sort_values(['revealed_ratings', 'model']).reset_index(drop=True)


def validate_sweep_weights(weights):
    """Validate a fixed, reproducible set of popularity mixing coefficients."""
    if weights is None:
        return None
    checked = tuple(float(w) for w in weights)
    if not checked or any(not np.isfinite(w) or w < 0 or w > 1 for w in checked):
        raise ValueError('Sweep weights must be nonempty finite values in [0, 1]')
    if len(set(checked)) != len(checked):
        raise ValueError('Sweep weights must not contain duplicates')
    return checked


def popularity_prior(model):
    """Training-only logarithmic popularity on [0, 1]."""
    counts = np.asarray(model.item_counts, dtype=np.float64)
    denominator = np.log1p(counts.max()) if counts.size else 0.0
    return (np.log1p(counts) / denominator if denominator > 0
            else np.zeros(len(model.movies), dtype=np.float64))


def popularity_sweep_scores(model, history, weights, *, top_k=3):
    """Score each candidate once with Top-K; blend with the same prior for all weights.

    At zero positive support, the Top-K component is the model's existing content
    fallback. The weight=1 score is always pure popularity. Returned scores are
    ranking heuristics on [0.5, 5], NOT calibrated rating estimates.
    """
    weights = validate_sweep_weights(weights)
    if top_k < 1:
        raise ValueError('top_k must be >= 1')
    # This is the expensive sparse multiplication and runs once per user/cap.
    base = model.score_topk_content(history, top_k=top_k, popularity_weight=0.0)
    similarity = np.clip((base.astype(np.float64) - 0.5) / 4.5, 0.0, 1.0)
    popularity = popularity_prior(model)
    return {
        f'topk_pop_w{weight:.4f}': np.asarray(
            0.5 + 4.5 * ((1 - weight) * similarity + weight * popularity),
            dtype=np.float32
        )
        for weight in weights
    }


def sweep_tables(ranking_rows, target_rows, weights):
    """Per-user and macro-average sweep ranking metrics; do not report heuristic RMSE."""
    weights = validate_sweep_weights(weights)
    prefix = 'topk_pop_w'
    ranking = pd.DataFrame(ranking_rows)
    sweep = ranking[ranking.model.str.startswith(prefix)].copy()
    if sweep.empty:
        return pd.DataFrame(), pd.DataFrame()
    sweep['popularity_weight'] = sweep.model.str[len(prefix):].astype(float)
    sweep['topk_weight'] = 1.0 - sweep['popularity_weight']
    columns = ['popularity_weight', 'topk_weight', 'revealed_ratings', 'user_id',
               'positive_targets', 'hit_at_10', 'hits_at_10', 'precision_at_10',
               'recall_at_10', 'ndcg_at_10']
    per_user = sweep[columns].sort_values(
        ['popularity_weight', 'revealed_ratings', 'user_id']).reset_index(drop=True)
    summary = (per_user.groupby(['popularity_weight', 'topk_weight', 'revealed_ratings'],
                                as_index=False)
               .agg(users=('user_id', 'nunique'),
                    hit_at_10=('hit_at_10', 'mean'),
                    precision_at_10=('precision_at_10', 'mean'),
                    recall_at_10=('recall_at_10', 'mean'),
                    ndcg_at_10=('ndcg_at_10', 'mean')))
    predictions = pd.DataFrame(target_rows)
    predictions = predictions[predictions.model.str.startswith(prefix)].copy()
    predictions['popularity_weight'] = predictions.model.str[len(prefix):].astype(float)
    median = (predictions.groupby(['popularity_weight', 'revealed_ratings'], as_index=False)
              .agg(median_target_rank=('rank', 'median')))
    summary = summary.merge(median, on=['popularity_weight', 'revealed_ratings'], how='left')
    return per_user, summary.sort_values(['revealed_ratings', 'popularity_weight']).reset_index(drop=True)


def plot_popularity_sweep(summary, output):
    """Compare NDCG@10 across visible histories, one line per popularity weight."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for weight, group in summary.groupby('popularity_weight', sort=True):
        ax.plot(group.revealed_ratings, group.ndcg_at_10,
                marker='o', label=f'Popularity {weight:.0%}')
    ax.set_xlabel('Ratings revealed to user profile')
    ax.set_ylabel('NDCG@10 (higher is better)')
    ax.set_title('Top-K popularity sweep (same held-out users and candidates)')
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def user_experiment(movies, ratings, *, users=50, histories=(0, 1, 3, 5, 10, 20),
                    holdout_positives=5, factors=40, negatives=200, full=False,
                    seed=42, return_rankings=False, compare_topk=False,
                    top_k=3, popularity_weight=0.15, sweep_weights=None):
    sweep_weights = validate_sweep_weights(sweep_weights)
    if holdout_positives < 1:
        raise ValueError('holdout_positives must be >= 1')
    rng = np.random.default_rng(seed)
    ratings = prepare(movies, ratings)
    max_history = max(histories)
    # A user is eligible only when there are at least max_history older ratings
    # before the earliest of their last N positive targets.
    eligible = []
    user_splits = {}
    for uid, group in ratings.groupby('userId', sort=False):
        pos_indices = np.flatnonzero(group.rating.to_numpy() >= 4.0)
        if len(pos_indices) < holdout_positives:
            continue
        last_positions = pos_indices[-holdout_positives:]
        cutoff = int(last_positions[0])
        if cutoff < max_history:
            continue
        eligible.append(int(uid))
        user_splits[int(uid)] = (group.iloc[:cutoff], group.iloc[last_positions], group)
    if not eligible:
        raise ValueError('No users have enough earlier ratings and positive targets')
    chosen = rng.choice(eligible, size=min(users, len(eligible)), replace=False)
    train = ratings.loc[~ratings.userId.isin(chosen)].copy()
    print(f'User experiment: fit on {len(train):,} ratings; '
          f'{len(chosen)} users completely excluded; {holdout_positives} targets each')
    model = HybridRecommender(factors=factors, random_state=seed).fit(movies, train)
    target_rows, ranking_rows = [], []
    for uid in chosen:
        prior, targets, full_history = user_splits[int(uid)]
        target_pairs = [
            (model.movie_to_tmdb[int(row.movieId)], float(row.rating))
            for row in targets.itertuples(index=False)
            if int(row.movieId) in model.movie_to_tmdb
        ]
        if len(target_pairs) != holdout_positives:
            continue
        target_ids = [mid for mid, _ in target_pairs]
        excluded_ids = {
            model.movie_to_tmdb[int(mid)] for mid in full_history.movieId
            if int(mid) in model.movie_to_tmdb
        }
        for cap in histories:
            # Reveal the most recent 'cap' ratings prior to the first target.
            support = prior.tail(cap) if cap else prior.iloc[:0]
            additional = {
                model.movie_to_tmdb[int(row.movieId)]: float(row.rating)
                for row in support.itertuples(index=False)
                if int(row.movieId) in model.movie_to_tmdb
            }
            components, history, _ = model.score_components(additional_ratings=additional)
            extra = None
            if compare_topk:
                extra = {
                    'topk': model.score_topk_content(history, top_k=top_k,
                                                     popularity_weight=0.0),
                    'topk_pop': model.score_topk_content(history, top_k=top_k,
                                                         popularity_weight=popularity_weight),
                }
            if sweep_weights is not None:
                extra = {**(extra or {}), **popularity_sweep_scores(
                    model, history, sweep_weights, top_k=top_k)}
            user_ranks, preds = evaluate_user_targets(
                model, target_ids, history, components, excluded_ids=excluded_ids,
                negatives=negatives, full=full,
                rng=np.random.default_rng(seed + int(uid)),
                extra_components=extra,
            )
            for mode, metrics in user_ranks.items():
                ranking_rows.append({
                    'user_id': int(uid), 'revealed_ratings': int(cap),
                    'model': mode, 'positive_targets': len(target_ids), **metrics,
                })
            actual_by_id = dict(target_pairs)
            for item in preds:
                target_rows.append({
                    'user_id': int(uid), 'revealed_ratings': int(cap),
                    'actual': actual_by_id[item['tmdb_id']], **item,
                })
    if not target_rows:
        raise ValueError('No valid user test targets selected')
    raw = pd.DataFrame(target_rows)
    rankings = pd.DataFrame(ranking_rows)
    summary = summarize_user_experiment(raw, rankings)
    if return_rankings:
        return raw, summary, rankings
    return raw, summary


def movie_experiment(movies, ratings, *, movie_count=20, caps=(0, 1, 5, 20),
                     min_interactions=40, max_test_per_movie=10,
                     factors=40, negatives=200, full=False, seed=42):
    rng = np.random.default_rng(seed)
    ratings = prepare(movies, ratings)
    counts = ratings.groupby('movieId').size()
    eligible = counts[counts >= max(min_interactions, max(caps) + max_test_per_movie)].index.to_numpy()
    if len(eligible) == 0:
        raise ValueError('No movies have sufficient interactions for the requested caps')
    selected = rng.choice(eligible, size=min(movie_count, len(eligible)), replace=False)
    is_selected = ratings.movieId.isin(selected)
    unaffected = ratings.loc[~is_selected]
    # Fixed positive test interactions for every cap, chosen after first max(caps) ratings.
    selected_train = {}
    tests = []
    for mid in selected:
        rows = ratings.loc[ratings.movieId == mid].sort_values(['timestamp', 'userId'])
        available = rows.iloc[max(caps):]
        positives = available[available.rating >= 4.0]
        test_rows = positives.tail(max_test_per_movie)
        if test_rows.empty:
            continue
        selected_train[int(mid)] = rows
        tests.append(test_rows)
    if not tests:
        raise ValueError('No positive test interactions for selected movies')
    test = pd.concat(tests, ignore_index=True)
    results = []
    for cap in caps:
        kept = [rows.iloc[:cap] for rows in selected_train.values() if cap]
        train = pd.concat([unaffected] + kept, ignore_index=True)
        print(f'Movie experiment cap={cap}: fitting on {len(train):,} ratings')
        model = HybridRecommender(factors=factors, random_state=seed).fit(movies, train)
        for row in test.itertuples(index=False):
            tid = model.movie_to_tmdb.get(int(row.movieId))
            if tid is None:
                continue
            uid = int(row.userId)
            components, history, _ = model.score_components(user_id=uid)
            # If an interaction for this target was included among the cap, do not test it.
            if tid in history:
                continue
            outcome = evaluate_target(model, tid, history, components, user_id=uid,
                                      negatives=negatives, full=full,
                                      rng=np.random.default_rng(seed + uid*1000003 + int(row.movieId)))
            results.append({'movie_id': int(row.movieId), 'user_id': uid,
                            'movie_training_ratings': int(cap), 'actual': float(row.rating), **outcome})
    if not results:
        raise ValueError('No movie test interactions remain')
    raw = pd.DataFrame(results)
    return raw, summarize(raw, 'movie_training_ratings')


def plot_summary(summary, x_col, path):
    """Plot cold-start ranking quality; no style/colors forced."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, data in summary.groupby('model', sort=False):
        ax.plot(data[x_col], data['ndcg_at_10'], marker='o', label=mode)
    ax.set_xlabel('Ratings available to model')
    ax.set_ylabel('NDCG@10 (higher is better)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/processed'))
    p.add_argument('--output', type=Path, default=Path('results/cold_start'))
    p.add_argument('--experiment', choices=('user', 'movie', 'both'), default='user')
    p.add_argument('--users', type=int, default=50)
    p.add_argument('--holdout-positives', type=int, default=5,
                   help='Number of latest positive (>=4 stars) movies held out per user')
    p.add_argument('--movies', type=int, default=20)
    p.add_argument('--factors', type=int, default=40)
    p.add_argument('--negatives', type=int, default=200)
    p.add_argument('--full', action='store_true', help='Use full catalog rather than sampled negatives')
    p.add_argument('--no-charts', action='store_true')
    p.add_argument('--compare-topk', action='store_true',
                   help='Also evaluate Top-K TF-IDF and Top-K with popularity prior for new users')
    p.add_argument('--top-k', type=int, default=3,
                   help='Number of nearest liked movies to average')
    p.add_argument('--popularity-weight', type=float, default=0.15,
                   help='Weight of training-only log popularity in Top-K scoring (0..1)')
    p.add_argument('--sweep-popularity-weights', type=float, nargs='+', default=None,
                   metavar='WEIGHT',
                   help='Top-K popularity mixing weights (0 to 1); 1 is popularity-only')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.sweep_popularity_weights is not None:
        try:
            args.sweep_popularity_weights = validate_sweep_weights(args.sweep_popularity_weights)
        except ValueError as exc:
            p.error(str(exc))
        if args.experiment == 'movie':
            p.error('--sweep-popularity-weights is supported for user experiments only')
    movies = pd.read_csv(args.data / 'training_movies.csv', low_memory=False)
    ratings = pd.read_csv(args.data / 'training_ratings.csv')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.experiment in ('user', 'both'):
        raw, summary, rankings = user_experiment(
            movies, ratings, users=args.users, factors=args.factors,
            negatives=args.negatives, full=args.full, seed=args.seed,
            holdout_positives=args.holdout_positives, return_rankings=True,
            compare_topk=args.compare_topk, top_k=args.top_k,
            popularity_weight=args.popularity_weight,
            sweep_weights=args.sweep_popularity_weights)
        raw.to_csv(args.output / 'user_examples.csv', index=False)
        rankings.to_csv(args.output / 'user_ranking.csv', index=False)
        summary.to_csv(args.output / 'user_summary.csv', index=False)
        if args.sweep_popularity_weights is not None:
            per_user, sweep_summary = sweep_tables(rankings, raw, args.sweep_popularity_weights)
            per_user.to_csv(args.output / 'popularity_sweep_users.csv', index=False)
            sweep_summary.to_csv(args.output / 'popularity_sweep.csv', index=False)
            if not args.no_charts:
                plot_popularity_sweep(sweep_summary, args.output / 'popularity_sweep_ndcg.png')
            print('\nPopularity sweep:\n', sweep_summary.round(4).to_string(index=False))
        if not args.no_charts:
            plot_summary(summary, 'revealed_ratings', args.output / 'user_ndcg_at_10.png')
        print('\nUser cold start:\n', summary.round(4).to_string(index=False))
    if args.experiment in ('movie', 'both'):
        raw, summary = movie_experiment(movies, ratings, movie_count=args.movies, factors=args.factors,
                                        negatives=args.negatives, full=args.full, seed=args.seed)
        raw.to_csv(args.output / 'movie_examples.csv', index=False)
        summary.to_csv(args.output / 'movie_summary.csv', index=False)
        if not args.no_charts:
            plot_summary(summary, 'movie_training_ratings', args.output / 'movie_ndcg_at_10.png')
        print('\nMovie cold start:\n', summary.round(4).to_string(index=False))
    print('Saved results in', args.output)


if __name__ == '__main__':
    main()
