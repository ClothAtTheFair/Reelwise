# Hybrid movie recommender — TMDB + MovieLens

Uses MovieLens **individual user ratings** for collaborative filtering and TMDB descriptions, genres, tags/keywords for content-based filtering. Movies with no MovieLens ratings are still eligible for content recommendations.

## Setup

1. Download the [TMDB Kaggle dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) and the [MovieLens dataset](https://grouplens.org/datasets/movielens/). Unzip both.
2. Put these files in `data/raw/`:
   - `TMDB_movie_dataset_v11.csv` (or specify another filename with `--tmdb`)
   - `ratings.csv` (MovieLens)
   - `links.csv` (MovieLens)
3. Install Python 3.10+ and dependencies: `pip install -r requirements.txt`.

## Commands (run at project root)

```bash
python filter_catalog.py --size 25000 --min-ratings 5
python train.py
python demo.py --user 1 --top 10
python demo.py --rate 27205:5 --rate 155:4.5 --top 10
python -m pytest -q
```

`filter_catalog.py` saves `data/processed/training_movies.csv` and `training_ratings.csv`; `train.py` evaluates on user-latest held-out ratings then saves `models/hybrid.joblib`. Model fit does **not** use the held-out ratings. Import `HybridRecommender` from `recommender.py` in a future FastAPI service.

## ID conventions

- Catalog and content model index are **TMDB IDs** (`id`).
- MovieLens ratings use **MovieLens `movieId`** values, mapped via `links.csv` `tmdbId`.
- `recommend(additional_ratings={...})` expects **TMDB IDs** unless `movie_id_type='movielens'` is passed.
- Existing user IDs in `recommend(user_id=...)` refer to MovieLens dataset IDs, not future application account IDs.

## Caveats / good academic practice

- This is an interpretable **baseline**. The collaborative model uses sparse user-mean-centered ratings and TruncatedSVD; predictions are heuristic rather than calibrated unbiased estimates, especially for sparse users. Content scores derive from signed preferences (rating minus 3.0) and TF-IDF cosine similarity.
- The blend down-weights collaborative predictions when the user or movie has little interaction data. New users can supply temporary explicit ratings, and TMDB-only movies use content scores.
- A new user without ratings gets a Bayesian-smoothed TMDB aggregate-vote fallback instead of personalized recommendations. Provide onboarding ratings for personalization.
- Filtering on full-dataset rating counts **before** the train/test split leaks popularity/availability information. The script is good for initial catalog creation; for publishable/strict evaluation, split raw MovieLens ratings first, select catalog using **training data only**, then intersect held-out ratings with the frozen catalog, and report dropped users/items. The evaluation script recomputes training counts but cannot undo an earlier selection leak.
- Held-out latest ratings yield MAE/RMSE. Sampled-negative Precision@10/Recall@10/NDCG@10 are *sampled-candidate* diagnostics, not full-catalog accuracy estimates; a single held-out positive per eligible user limits interpretability.
- For reliable benchmarking, compare collaborative-only, content-only, hybrid, and popularity baselines on the same split; tune hyperparameters on a separate validation set, then test once.
- 25k movies is a starting point, not a required size. A compact corpus is much faster to train. No NxN dense movie similarity matrix is created.
- joblib deserialization can execute arbitrary code. Only load trusted model artifacts.
- Verify dataset licensing and attribution for published frontend use, poster display, and redistribution.

## FastAPI backend (new)

The backend stores app accounts and ratings in **SQLite**, separately from the
MovieLens training data. Authenticated user IDs are **not MovieLens IDs**.

```bash
# After filter_catalog.py and train.py:
uvicorn backend:app --reload
# Interactive API documentation: http://127.0.0.1:8000/docs
```

Settings (optional environment variables): `MODEL_PATH`, `APP_DB_PATH`, and
`CORS_ORIGINS` (comma-separated allowed frontend origins; defaults to localhost
ports 5173 and 3000). Start the API only after training. You can also inspect
`GET /health` without a model, but model-based endpoints return 503 until trained.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/users` | Create a user; returns a **one-time bearer token** |
| GET | `/users/me` | View signed-in user's profile |
| GET | `/movies?q=galaxy&limit=20&offset=0` | Search movies by title |
| GET | `/movies/{tmdb_id}` | Get movie metadata |
| GET | `/ratings` | Get signed-in user's movie ratings |
| PUT | `/ratings` | Save or update a `{ "tmdb_id": 27205, "rating": 4.5 }` rating |
| DELETE | `/ratings/{tmdb_id}` | Delete user's rating |
| GET | `/recommendations?top_n=10` | Personalized recommendations excluding rated movies |
| GET | `/health` | Model readiness |

Endpoints for ratings and personalized recommendations require the header
`Authorization: Bearer YOUR_TOKEN`. Example:

```bash
curl -X POST http://localhost:8000/users -H 'Content-Type: application/json' -d '{"username":"demo_student"}'
# Copy the token from the response:
curl http://localhost:8000/movies?q=Inception
curl -X PUT http://localhost:8000/ratings -H 'Authorization: Bearer YOUR_TOKEN' \
  -H 'Content-Type: application/json' -d '{"tmdb_id":27205,"rating":5}'
curl http://localhost:8000/recommendations?top_n=10 -H 'Authorization: Bearer YOUR_TOKEN'
```

### How new app ratings affect recommendations

New app ratings affect content-based recommendations **immediately**. To train
collaborative factors for app users, periodically rerun:

```bash
python train.py --app-db data/app.sqlite3
```

This retrains the *whole* saved model from processed MovieLens ratings plus app
ratings for TMDB movies with valid MovieLens links, then the API notices the
updated model file and loads it on the next request. TMDB-only movies remain
content-based. Evaluation is calculated on MovieLens holdout ratings *before*
including app-user interactions. For classroom use, retrain manually rather
than doing so for every rating.

### Security / deployment notes

This is a **school project backend**, not a production identity service. Tokens
are generated once, stored hashed, and returned only on registration; users must
store their token safely. There is no password login, token recovery/revocation,
rate limiting, or administrative tooling. Use HTTPS, configure `CORS_ORIGINS`,
and add full authentication and abuse prevention before any public deployment.
Do not commit `data/app.sqlite3`, issued tokens, or exported user ratings.
Also load only locally trained trusted `.joblib` files (pickle-based loading).

## Genre preferences (new)

Users can select preferred TMDB genres independently of movie ratings. Genre choices are stored in SQLite in `user_genre_preferences` and apply immediately without retraining. The table is automatically created on API startup for existing databases. Supported genres are the standard TMDB genre names; `Sci-Fi` is accepted as an alias for `Science Fiction`.

With the bearer token returned by `POST /users`, send:

```http
PUT /users/me/genres
Authorization: Bearer YOUR_TOKEN
Content-Type: application/json

{"genres": ["Action", "Science Fiction", "Thriller"]}
```

`GET /users/me/genres` returns `{ "genres": [...] }`. `PUT` **replaces** all saved genre choices. Pass `{"genres": []}` to clear them. Unknown genres return `422`, and duplicate/case-variant genres are normalized. Both routes require authentication.

`GET /recommendations` now uses these choices and returns `preferred_genres` plus a `genre_match` (0–1) on each movie. The recommender combines its prior score and genre match after normalizing the prior 0.5–5 score to 0–1:

```
final_score = 0.5 + 4.5 * (0.75 * ((hybrid_score - 0.5) / 4.5) + 0.25 * genre_match)
```

The 25% genre contribution is a tunable heuristic. No genre choices means the original scores are unchanged. Genre rescoring runs on **all** eligible movies before top-N selection (not just a preselected shortlist). Explicit genre choices are distinct from implicit preferences already inferred from rated movie content. For full academic evaluation, tune the weight using held-out interactions.

Run tests with `python -m pytest -q` (the tiny synthetic SVD dataset may cause harmless explained-variance warnings).

## Compare baseline, collaborative, content and hybrid models

`evaluate.py` is an **offline evaluation tool**. It fits an evaluation-only
`HybridRecommender` on the temporal training split and scores *the same held-out
users and ranking candidate movies* using four prediction modes:

- `baseline`: shrinkage-adjusted training-only user and movie means
- `collaborative`: mean-centered Truncated SVD estimate
- `content`: signed TF-IDF profile estimate
- `hybrid`: dynamically weighted collaborative/content estimate

It never overwrites `models/hybrid.joblib` or changes the running FastAPI model.

```bash
pip install -r requirements.txt
python evaluate.py --max-rating-users 2000 --ranking-users 100 --candidates 200
# More realistic, more expensive ranking among all available eligible movies:
python evaluate.py --ranking-mode full --ranking-users 100 --max-rating-users 2000
# Rating accuracy on every eligible held-out user (can be slow):
python evaluate.py --max-rating-users 0 --ranking-users 100
```

The script saves `results/model_comparison.csv`, `results/ndcg_at_k.csv`,
`results/performance_by_user_activity.csv`, `results/evaluation_metadata.json`,
and three PNG figures. Increase the sample size if your machine can handle it.
A fixed `--seed` makes the split and sampled ranking candidates reproducible.
Use the *same* candidate sets for all models in each run.

The `HybridRecommender.predict(..., mode='baseline'|'collaborative'|'content'|'hybrid')`
method exposes component predictions. The existing FastAPI interface and
`recommend()` behavior are unchanged. `train.py` also accepts `--seed` and uses
the same `temporal_split()` helper as `evaluate.py`.

**Interpret results carefully:** The holdout contains one latest rating per
eligible user. For ranking, only held-out ratings of at least four stars count
as positives; unobserved movies are *assumed* negatives and may actually be
liked. With one known positive per user, Precision@10 can be at most 0.1.
Sampled-negative ranking (default) is easier than full-catalog ranking and
should be labeled as such. Genre preferences are not evaluated because public
MovieLens ratings have no explicitly selected genre preferences. Most
importantly, `filter_catalog.py` may have selected movies based on *all* ratings
before the temporal split. To eliminate selection leakage for a final academic
benchmark, perform the raw-rating holdout before choosing the training catalog.

## Fixed hybrid-weight sweep and cold-start analysis

`evaluate.py` now supports a fixed-weight sweep **without separately refitting the
model for each weight**. The fixed weight is the collaborative fraction; the
content fraction is `1 - weight`. The sweep falls back to content for movies
with **zero training ratings** or for users not in the fitted SVD.

```bash
python evaluate.py --ranking-mode full --ranking-users 100 --max-rating-users 2000 \
  --sweep-weights 0 0.25 0.5 0.75 0.85 0.9 0.95 1
```

New output files in `results/`:

- `hybrid_weight_sweep.csv`: MAE, RMSE, Precision@10, Recall@10, NDCG@10 by fixed collaborative weight.
- `cold_start_comparison.csv`: same measures by **training-only** movie rating counts (`0`, `1-4`, `5-19`, `20+`) and user training history (`1-9`, `10-49`, `50-199`, `200+`), for each original model and sweep variant.
- `hybrid_weight_rmse.png`, `hybrid_weight_ndcg.png`, `cold_start_movie_rmse.png`.

The existing `hybrid` row continues to use the model's **adaptive** weighting;
`fixed_0.850` in the cold-start CSV represents a fixed 85% collaborative blend
when collaborative information exists. A fixed weight of 1.0 has content fallback
for unrated movies; consequently it is not identical to the standalone
`collaborative` mode for these cold-start movies.

Only movies with a held-out relevant interaction can be assessed; a zero-count
movie bucket may be empty, depending on the holdout. The leave-last-one-out
protocol contains only one relevant held-out movie per user, so full-catalog
Recall@10 is a *hit rate* and Precision@10 cannot exceed 0.1. Unrated candidates
are assumed negative, not verified negative. The processed catalog may have
been selected using all ratings, which may leak held-out interactions into
catalog membership; split the raw interactions before catalog selection for
strict evaluation. **Do not choose the best weight using the same held-out set
that you report as your final test metric**. Use a validation set for the sweep
and an untouched test set for final reported metrics.

## Controlled cold-start experiments

The new `cold_start_experiments.py` evaluates **deliberately restricted** user and movie histories, rather than only bucketing normal held-out ratings. It produces per-example CSVs, aggregated CSVs, and NDCG@10 plots under `results/cold_start/`.

Quick user experiment (one SVD fit):

```bash
python cold_start_experiments.py --experiment user --users 50
```

Movie experiment (refits SVD at each movie interaction cap 0, 1, 5, 20; significantly more expensive):

```bash
python cold_start_experiments.py --experiment movie --movies 20
```

Full-catalog evaluation rather than 200 sampled assumed negatives:

```bash
python cold_start_experiments.py --experiment user --users 100 --full
```

The user experiment completely excludes selected test users from model fitting and reveals their earlier ratings only at prediction time. The *current* `HybridRecommender` does **not** support collaborative factor fold-in for unseen users; thus collaborative scores for them are a fallback, not personalized collaborative filtering. This deliberately measures the currently deployed new-user behavior. The movie experiment reserves the same held-out interactions for each interaction cap and refits the model per cap. The `hybrid90` mode uses 90% collaborative scoring *only when a fitted user factor and an item rating history exist*, otherwise content is used.

**Important caveats:** The processed catalog could already have been filtered using all MovieLens ratings, so the experiment is not fully leakage-free at catalog-selection level. All ranking metrics rely on incomplete implicit relevance labels: movies without user ratings are treated as assumed negatives. The user experiment reserves multiple positive targets per user (default five), drawn from their latest positively rated movies; all revealed ratings precede the earliest test target. For stronger evidence, repeat with multiple seeds, use a separate tuning/validation cohort, report confidence intervals, and evaluate explicit genre onboarding separately. The movie experiment's rating examples may contain multiple users/movies, so individual observations are not independent for simple confidence intervals. The 0, 1, 5, 20 movie groups represent *actual controlled training interaction caps*, not bins from the standard holdout evaluation.

### Updated multi-positive user cold-start experiment (November 2026)

`cold_start_experiments.py` now reserves the **last five positive movies** (ratings of at least 4 stars) for each test user, subject to having enough preceding history. The users are *entirely absent* from model training. For each user, 0, 1, 3, 5, 10, and 20 preceding ratings are revealed (most recent first); the same five test movies and the same sampled candidates are used across all reveal conditions. **No held-out target ratings enter the content profile or collaborative fit.**

```bash
python cold_start_experiments.py --experiment user --users 100 --holdout-positives 5
# More realistic but slower: rank against the entire eligible catalog
python cold_start_experiments.py --experiment user --users 100 --holdout-positives 5 --full
```

Output files:
- `user_examples.csv`: each held-out rating, its prediction, and its rank for each model/reveal condition.
- `user_ranking.csv`: one row per user, reveal condition and scoring model, containing *multi-positive* Hit@10, Precision@10, Recall@10 and NDCG@10.
- `user_summary.csv`: macro-averaged ranking metrics across users and MAE/RMSE over target ratings, plus median target rank.
- `user_ndcg_at_10.png`: mean per-user NDCG@10 by number of revealed ratings.

The recommender's TF-IDF user profile now weights rated movies by `rating - mean(visible user ratings)`, replacing the constant `rating - 3`. It also changes normal live content-based personalization. With just **one rating**, or all ratings equal, all centered weights are zero and the model falls back to its existing nonpersonalized TMDB-vote score; this is a limitation of strict mean centering. Test whether this improves your results rather than assuming it will. A multi-positive user test measures more than one relevant item per user, but unrated candidates are still only *assumed* negative, and a small 100-user sample can be noisy.

The movie cold-start experiment is unchanged in design; it will use the updated TF-IDF profile because that lives in `recommender.py`. The new mean-centering rule is implemented at inference time, so changing it does not mathematically require rebuilding the stored TF-IDF or SVD factors. Restart the API after updating `recommender.py`. For reproducible releases, you may also rebuild the saved model with the new code version.

## Top-K liked-movie similarity + modest popularity prior (experimental)

This optional user cold-start experiment compares the existing approaches against:

- `topk`: mean cosine similarity to the `--top-k` closest **visible positively rated** movies (rating >= 4), without a popularity boost.
- `topk_pop`: the same score mixed with `--popularity-weight` of log-scaled **training-only MovieLens rating counts**. Default 0.15 means 85% Top-K similarity and 15% popularity, after both are mapped to 0..1.

Use identical held-out targets and candidate pools for all methods:

```bash
python cold_start_experiments.py --experiment user --users 100 \
  --holdout-positives 5 --full --compare-topk --top-k 3 --popularity-weight 0.15
```

For a quick test, omit `--full`. Outputs contain additional `topk` and `topk_pop` model rows in `user_summary.csv`, `user_ranking.csv`, and `user_examples.csv` plus corresponding curves in `user_ndcg_at_10.png`.

**How scoring works**: TF-IDF vectors are L2-normalized. Each catalog candidate gets its average of the three highest cosine similarities with rated >=4 movies. Rating counts are compressed as `log1p(count)/log1p(max_count)` using only fitted training interactions. Final ranking score = `0.85 * topk_similarity + 0.15 * popularity`, rescaled to 0.5..5. This output is a **ranking heuristic**, not a calibrated star prediction; treat NDCG/Recall as the primary comparison metrics and interpret RMSE for these experimental modes cautiously.

The `topk` implementation computes sparse candidate-to-liked similarities in chunks, without generating an NxN item similarity matrix. At zero positive ratings both new methods use the original metadata-based cold-start fallback to keep that condition comparable. As in the rest of the controlled-user experiment, selected evaluation users are excluded entirely from SVD fitting, so `hybrid90` equals `content` for those users. The new methods do not change FastAPI's default recommendations or the saved model's training process. The standard TF-IDF profile was set to positive-only weighting (`max(rating - 3, 0)`) to match the latest experiment discussed.

**Caution**: The ranking candidates and historical rating counts in the original filtered catalog may still have selection bias from pre-filtering, and unobserved movies are only *assumed* negatives. To draw final conclusions, use a training-only catalog selection and repeat the experiment across several random seeds.


## Top-K popularity parameter sweep

Use `--sweep-popularity-weights` with the **user** cold-start experiment.
The parameter is the fraction of the final ranking score supplied by training-only
log popularity, with the balance supplied by Top-K positive-movie similarity.
Weight `0` is Top-K alone; weight `1` is a **popularity-only baseline**.

```bash
python cold_start_experiments.py --experiment user --users 100 \
  --holdout-positives 5 --top-k 3 --full \
  --sweep-popularity-weights 0 0.05 0.10 0.15 0.20 0.25 1.0
```

Optionally add `--compare-topk` for the original Top-K/Top-K+popularity
comparison in the ordinary summary, or use `--no-charts` to skip PNG output.
The sweep still uses the same sampled users, held-out targets, and candidate
sets for **every weight** and **every revealed-history level**. The recommender
fits once; per-user/cap Top-K similarities are calculated once and blended for
all weights without fitting a new model.

Output in `results/cold_start/`:

- `popularity_sweep.csv`: macro-averaged NDCG@10, Recall@10, Hit@10,
  Precision@10 and median held-out movie rank by history length and weight.
- `popularity_sweep_users.csv`: per-user matched ranking metrics for paired
  comparisons and bootstrap confidence intervals.
- `popularity_sweep_ndcg.png`: NDCG@10 against revealed rating count for
  each popularity weight.

**Important:** This experiment evaluates *ranking* scores, not calibrated
star predictions. The general `user_summary.csv` also reports MAE/RMSE of
these scores, but do not use the sweep's MAE/RMSE to select weights.
At 0 positive ratings, the Top-K portion uses the existing content cold-start
fallback; the popularity-only weight uses training rating counts. Retain an
untouched user group for final testing after choosing hyperparameters to avoid
optimistic selection bias. Catalog filtering may still reflect interaction
counts from the entire raw dataset; strict leakage-free research requires
creating the catalog with training interactions only.

## Frontend-ready production ranking (v2)

The production `GET /recommendations` now uses the best *experimental starting
configurations* from the cold-start studies, without changing the independent
`evaluate.py` / `cold_start_experiments.py` benchmarks:

| User situation | Ranking method |
|---|---|
| No positively rated movies | Training-set log-popularity; selected genres can boost matches |
| New app user, at least one rating of >=4 stars | Top-K TF-IDF (K=10) 50% + log-popularity 50% |
| App user included in an up-to-date SVD retrain | 90% collaborative + 10% Top-K/popularity ranking for rated-in-training movies |
| Movie with no SVD training ratings | Top-K/popularity content-only fallback |

The selected genres are applied to the resulting scores at **25%** of the
final normalized ranking, when supplied. Movies already rated by the user are
excluded. The blend is a **ranking heuristic**, not a predicted star rating.
Predicted-star evaluation still uses `score_components()` / `predict()` and is
unchanged; the original `evaluate.py` numbers are not a measurement of this new
production ranking. Validate the updated production policy independently before
making academic performance claims.

Top-K uses the most similar positively rated movies; if fewer than K exist,
all available liked movies are used. Without positively rated movies, the
production ranking falls back to a consistent popularity-only rule. The
popularity prior is based on training-set interaction counts (not held-out
ratings). TMDB-only movies can still be recommended through content or genre
matches, though a strong popularity prior favors movies with training history.

### Frontend endpoints

| Method | Path | Authentication |
|---|---|---|
| GET | `/health` | No |
| GET | `/genres` | No; lists genre checkbox choices |
| POST | `/users` | No; returns a one-time bearer token |
| GET | `/users/me` | Bearer token |
| GET | `/users/me/genres` | Bearer token |
| PUT | `/users/me/genres` | Bearer token; body `{"genres":["Action","Science Fiction"]}` |
| GET | `/movies?q=space&limit=20&offset=0` | No |
| GET | `/movies/{tmdb_id}` | No |
| GET | `/ratings` | Bearer token |
| PUT | `/ratings` | Bearer token; body `{"tmdb_id":10,"rating":4.5}` |
| DELETE | `/ratings/{tmdb_id}` | Bearer token |
| GET | `/recommendations?top_n=10` | Bearer token |

Recommendation payload fields:

- `items`: movie metadata (`tmdb_id`, `title`, `genres`, `overview`, `poster_path`,
  `release_date`), `score`, `collaborative_weight`, `genre_match`, `strategy`.
- `recommendation_strategy`: `popularity`, `topk_popularity`, or
  `collaborative_topk_hybrid`.
- `scoring`: configuration and explicit notice that scores are *ranking*
  scores, **not star predictions**.
- `rating_count`, `positive_rating_count`, `preferred_genres`.

CORS defaults to development frontends on `http://localhost:5173` and
`http://localhost:3000`. If the frontend runs at another origin, set
`CORS_ORIGINS` to its exact comma-separated origins before starting Uvicorn.
The starter authentication returns a one-time bearer token at registration,
with no password login/recovery; it is for classroom/local development, not a
public production deployment. Use HTTPS and a proper authentication provider
for a public site. Only load trusted `joblib` model files.

### Retraining app users

```bash
python train.py --app-db data/app.sqlite3 --skip-evaluation
```

The API reloads the model artifact when its file modification time changes.
The newly saved training artifact includes an app-rating snapshot **including
TMDB-only ratings**, so the backend can detect whether the user's SVD factors
are current. If ratings have changed since the last model fit, the API uses
Top-K/popularity until the next retrain. New ratings immediately affect Top-K
and genre preferences immediately affect ranking. Existing model artifacts
still load without retraining, but you need a retrain to add application users
into collaborative SVD.

### Test and run

```bash
python -m pytest -q
uvicorn backend:app --reload
```

The automatically generated documentation is at
`http://127.0.0.1:8000/docs`.

## Automated batch retraining

Run `python batch_retrain.py --daemon --interval-hours 24` alongside FastAPI, or schedule `python batch_retrain.py --once` daily using Windows Task Scheduler or cron. Training is skipped when no app ratings changed. Completed models replace `models/hybrid.joblib` atomically and the API auto-reloads on its next request. See the project root README for setup and failure recovery.
