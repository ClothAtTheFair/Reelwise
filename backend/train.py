"""Evaluate on held-out latest ratings, then refit and save full-data model."""
import argparse
import sqlite3
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from recommender import HybridRecommender


def temporal_split(ratings, min_user_ratings=5):
    """Deterministic leave-last-one-out holdout shared with evaluate.py."""
    ratings = ratings.copy()
    if 'timestamp' not in ratings:
        raise ValueError('ratings.csv must have timestamps for temporal holdout')
    ratings = ratings.sort_values(['userId', 'timestamp', 'movieId'])
    counts = ratings.groupby('userId').movieId.transform('size')
    eligible = counts >= min_user_ratings
    holdout_index = ratings[eligible].groupby('userId').tail(1).index
    return ratings.drop(index=holdout_index).copy(), ratings.loc[holdout_index].copy()


def evaluate(model, test, max_ranking_users=100, candidate_negatives=200, seed=42):
    predictions, actuals = [], []
    for row in test.itertuples(index=False):
        if int(row.movieId) not in model.movie_to_tmdb:
            continue
        predictions.append(model.predict(row.userId, row.movieId, movie_id_type='movielens'))
        actuals.append(float(row.rating))
    if not predictions:
        return {'evaluated_ratings': 0}
    y = np.asarray(actuals)
    p = np.asarray(predictions)
    results = {'evaluated_ratings': len(y),
               'mae': round(float(np.mean(np.abs(y-p))), 4),
               'rmse': round(float(np.sqrt(np.mean((y-p)**2))), 4)}
    # Ranking: held-out positive >= 4 stars versus sampled unseen candidates.
    # Sampled-negative NDCG@10 is NOT a full-catalog ranking metric.
    rng = np.random.default_rng(seed)
    positives = test[test.rating >= 4].sample(frac=1, random_state=seed).head(max_ranking_users)
    ndcgs, recalls, precisions = [], [], []
    movie_ids = model.movies.id.to_numpy(dtype=np.int64)
    for row in positives.itertuples(index=False):
        target = model.movie_to_tmdb.get(int(row.movieId))
        if target is None:
            continue
        seen = set(model.user_ratings.get(int(row.userId), {}))
        pool = movie_ids[~np.isin(movie_ids, list(seen | {target}))]
        if not len(pool):
            continue
        sampled = rng.choice(pool, size=min(candidate_negatives, len(pool)), replace=False)
        candidates = np.append(sampled, target)
        scores, _, _ = model._scores(row.userId)
        candidate_scores = scores[[model.id_to_col[int(mid)] for mid in candidates]]
        ranking = candidates[np.argsort(-candidate_scores)]
        position = int(np.where(ranking == target)[0][0]) + 1
        ndcgs.append(1 / np.log2(position + 1) if position <= 10 else 0.0)
        recalls.append(float(position <= 10))
        precisions.append(float(position <= 10) / 10.0)
    results.update({'ranking_users': len(ndcgs),
                    'sampled_precision_at_10': round(float(np.mean(precisions)), 4) if ndcgs else None,
                    'sampled_recall_at_10': round(float(np.mean(recalls)), 4) if ndcgs else None,
                    'sampled_ndcg_at_10': round(float(np.mean(ndcgs)), 4) if ndcgs else None})
    return results


def read_app_ratings(db_path, movies):
    """Convert saved app-user TMDB ratings to MovieLens IDs for SVD training.

    Movies without a MovieLens link remain content-only; no link is fabricated.
    """
    if not db_path or not db_path.exists():
        return pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
    from backend import APP_USER_OFFSET
    with sqlite3.connect(db_path) as db:
        try:
            saved = pd.read_sql_query('SELECT user_id, tmdb_id, rating, updated_at FROM ratings', db)
        except pd.io.sql.DatabaseError:
            return pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
    if saved.empty:
        return pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])
    links = movies[['id', 'movieId']].dropna(subset=['movieId']).drop_duplicates('id')
    saved = saved.merge(links, left_on='tmdb_id', right_on='id', how='inner')
    saved['userId'] = saved['user_id'].astype('int64') + APP_USER_OFFSET
    saved['movieId'] = saved['movieId'].astype('int64')
    saved['timestamp'] = pd.to_datetime(saved['updated_at'], utc=True).astype('int64') // 10**9
    return saved[['userId', 'movieId', 'rating', 'timestamp']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/processed'))
    p.add_argument('--output', type=Path, default=Path('models/hybrid.joblib'))
    p.add_argument('--factors', type=int, default=40)
    p.add_argument('--seed', type=int, default=42, help='SVD random seed (matches evaluate.py)')
    p.add_argument('--skip-evaluation', action='store_true')
    p.add_argument('--app-db', type=Path, default=None, help='Optional SQLite app ratings for final training')
    args = p.parse_args()
    movies = pd.read_csv(args.data / 'training_movies.csv', low_memory=False)
    ratings = pd.read_csv(args.data / 'training_ratings.csv')
    if ratings.empty:
        raise ValueError('No ratings selected; verify MovieLens links and input paths')
    if not args.skip_evaluation:
        train, test = temporal_split(ratings)
        # Recalculate training counts; do not use held-out rating counts as features.
        train_counts = train.groupby('movieId').size()
        evaluation_movies = movies.copy()
        evaluation_movies['rating_count'] = evaluation_movies.movieId.map(train_counts).fillna(0).astype(int)
        print('Holdout evaluation (latest rating from users with >=5 ratings):')
        temp_model = HybridRecommender(factors=args.factors, random_state=args.seed).fit(evaluation_movies, train)
        print(evaluate(temp_model, test))
    print('Training final model on all selected ratings...')
    if args.app_db is not None:
        app_ratings = read_app_ratings(args.app_db, movies)
        print(f'Loaded {len(app_ratings):,} linked app ratings for final training')
        if not app_ratings.empty:
            ratings = pd.concat([ratings, app_ratings], ignore_index=True)
    model = HybridRecommender(factors=args.factors, random_state=args.seed).fit(movies, ratings)
    # Full app-rating snapshot also includes TMDB-only titles. This allows the
    # API to detect newly changed app ratings without accidentally discarding
    # valid SVD factors from users who rated TMDB-only titles before retraining.
    model.app_rating_snapshots = {}
    if args.app_db is not None and args.app_db.exists():
        from backend import APP_USER_OFFSET
        with sqlite3.connect(args.app_db) as db:
            try:
                rows = db.execute('SELECT user_id, tmdb_id, rating FROM ratings').fetchall()
            except sqlite3.OperationalError:
                rows = []
        for uid, tmdb, rating in rows:
            key = APP_USER_OFFSET + int(uid)
            model.app_rating_snapshots.setdefault(key, {})[int(tmdb)] = float(rating)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output, compress=3)
    print(f'Saved {args.output} with {len(model.movies):,} movies and {len(model.users):,} users')
    print('Only load model files from sources you trust (joblib uses pickle).')


if __name__ == '__main__':
    main()
