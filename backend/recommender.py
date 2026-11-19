"""Sparse hybrid recommender. No dense item-to-item similarity matrix.

SVD models residual ratings about each user's mean. Content uses a positive-only
TF-IDF preference profile, with optional Top-K similarity ranking. Scores are heuristic 0.5-5 star estimates;
calibration should be assessed on held-out interactions.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class HybridRecommender:
    def __init__(self, factors=40, max_features=20000, random_state=42):
        self.factors = factors
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, movies, ratings):
        movies = movies.copy().drop_duplicates('id').reset_index(drop=True)
        movies['id'] = pd.to_numeric(movies.id, errors='raise').astype('int64')
        if movies.empty:
            raise ValueError('No movies to train on')
        self.movies = movies
        self.id_to_col = {int(mid): i for i, mid in enumerate(movies.id)}
        self.movie_to_tmdb = {
            int(row.movieId): int(row.id)
            for row in movies[['movieId', 'id']].dropna().itertuples(index=False)
        } if 'movieId' in movies else {}
        self.genre = movies.get('genres', pd.Series('', index=movies.index)).fillna('').astype(str)
        def feature(col):
            return movies.get(col, pd.Series('', index=movies.index)).fillna('').astype(str)
        corpus = (feature('genres').str.replace(',', ' ', regex=False) + ' ' +
                  feature('genres').str.replace(',', ' ', regex=False) + ' ' +
                  feature('keywords').str.replace(',', ' ', regex=False) + ' ' +
                  feature('overview') + ' ' + feature('tagline'))
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=self.max_features,
                                          min_df=1, ngram_range=(1, 2), dtype=np.float32)
        self.features = self.vectorizer.fit_transform(corpus)
        ratings = ratings.copy()
        ratings = ratings[ratings.movieId.isin(self.movie_to_tmdb)].copy()
        ratings['tmdb_id'] = ratings.movieId.map(self.movie_to_tmdb)
        ratings = ratings.drop_duplicates(['userId', 'tmdb_id'], keep='last')
        self.user_ratings = {}
        self.user_means = {}
        self.user_counts = {}
        for uid, group in ratings.groupby('userId'):
            values = {int(tid): float(r) for tid, r in zip(group.tmdb_id, group.rating)}
            self.user_ratings[int(uid)] = values
            self.user_means[int(uid)] = float(np.mean(list(values.values())))
            self.user_counts[int(uid)] = len(values)
        self.item_counts = np.zeros(len(movies), dtype=np.int32)
        item_sums = np.zeros(len(movies), dtype=np.float64)
        for vals in self.user_ratings.values():
            for mid, rating in vals.items():
                col = self.id_to_col[mid]
                self.item_counts[col] += 1
                item_sums[col] += rating
        self.global_rating_mean = float(ratings.rating.mean()) if len(ratings) else 3.0
        self.item_rating_means = np.divide(
            item_sums, self.item_counts,
            out=np.full(len(movies), self.global_rating_mean, dtype=np.float64),
            where=self.item_counts > 0,
        )
        self.users = sorted(self.user_ratings)
        self.user_to_row = {uid: i for i, uid in enumerate(self.users)}
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        if len(self.users) >= 2 and len(movies) >= 2 and ratings.shape[0] > 0:
            rows, cols, data = [], [], []
            for uid, entries in self.user_ratings.items():
                for mid, score in entries.items():
                    rows.append(self.user_to_row[uid])
                    cols.append(self.id_to_col[mid])
                    data.append(score - self.user_means[uid])
            sparse = csr_matrix((data, (rows, cols)), shape=(len(self.users), len(movies)), dtype=np.float32)
            k = min(self.factors, min(sparse.shape) - 1)
            if k > 0 and sparse.nnz:
                self.svd = TruncatedSVD(n_components=k, random_state=self.random_state)
                self.user_factors = self.svd.fit_transform(sparse)
                self.item_factors = self.svd.components_.T
        return self

    def _resolve(self, movie_id, movie_id_type):
        if movie_id_type == 'tmdb':
            return int(movie_id)
        if movie_id_type == 'movielens':
            return self.movie_to_tmdb.get(int(movie_id))
        raise ValueError("movie_id_type must be 'tmdb' or 'movielens'")

    def _profile(self, history):
        mids = [self.id_to_col[mid] for mid in history if mid in self.id_to_col]
        if not mids:
            return None
        values = np.array([history[int(self.movies.iloc[i].id)] for i in mids], dtype=np.float32)
        # Positive-only preference evidence: ratings <= 3 do not contribute.
        weights = np.maximum(values - 3.0, 0.0)
        if not np.any(weights):
            return None
        profile = self.features[mids].multiply(weights[:, None]).sum(axis=0)
        profile = np.asarray(profile).reshape(1, -1)
        norm = float(np.linalg.norm(profile))
        return None if norm < 1e-9 else profile / norm

    def score_topk_content(self, history, *, top_k=3, popularity_weight=0.15,
                           min_rating=4.0, chunk_size=4096):
        """Score every movie by its Top-K similarity to positively rated movies.

        - Uses the *already fitted* L2-normalized sparse TF-IDF item vectors.
        - Each candidate is compared only to the visible liked movies, never all
          item pairs. Chunks cap peak memory for large catalogs.
        - A log-scaled popularity prior comes from TRAINING interaction counts;
          no test interactions are used in constructing the prior.
        - Returns 0.5..5 heuristic scores, intended for ranking; compare ranking
          metrics instead of assuming these values are calibrated star predictions.
        - When no positive support exists, use the existing cold-start fallback.
        """
        if top_k < 1 or chunk_size < 1:
            raise ValueError('top_k and chunk_size must be positive')
        if not 0.0 <= popularity_weight <= 1.0:
            raise ValueError('popularity_weight must be in [0, 1]')
        liked = [(self.id_to_col[int(mid)], float(rating))
                 for mid, rating in history.items()
                 if int(mid) in self.id_to_col and float(rating) >= min_rating]
        if not liked:
            # Identical cold-start ranking to the pre-existing content fallback.
            # This is computed from catalog metadata only, not evaluation targets.
            fallback, _, _ = self.score_components(additional_ratings=history)
            return np.asarray(fallback['content'], dtype=np.float32).copy()

        liked_cols = np.asarray([col for col, _ in liked], dtype=np.int64)
        liked_vectors = self.features[liked_cols]  # TF-IDF L2 normalizes rows
        n = len(self.movies)
        similarities = np.empty(n, dtype=np.float32)
        actual_k = min(int(top_k), len(liked))

        # Bound the temporary dense similarity block when a trained user has
        # thousands of positive ratings (e.g., extensive MovieLens histories).
        effective_chunk = min(chunk_size, max(32, 2_000_000 // len(liked)))
        for start in range(0, n, effective_chunk):
            end = min(start + effective_chunk, n)
            # Shape (chunk_size, number_of_visible_liked_movies), not (n,n).
            sim = (self.features[start:end] @ liked_vectors.T).toarray()
            np.maximum(sim, 0.0, out=sim)
            if actual_k == sim.shape[1]:
                similarities[start:end] = np.mean(sim, axis=1)
            else:
                # Mean of the K closest positively rated movies per candidate.
                largest = np.partition(sim, -actual_k, axis=1)[:, -actual_k:]
                similarities[start:end] = largest.mean(axis=1)

        # Training-set popularity [0, 1], compressed to avoid domination by hits.
        counts = self.item_counts.astype(np.float64)
        max_log = float(np.log1p(counts.max())) if counts.size else 0.0
        popularity = (np.log1p(counts) / max_log if max_log > 0
                      else np.zeros(n, dtype=np.float64))

        # Score against all movies on comparable [0,1] components. A small
        # popularity term breaks near-ties without overwhelming similarity.
        ranking = ((1.0 - popularity_weight) * similarities
                   + popularity_weight * popularity)
        return np.clip(0.5 + 4.5 * ranking, 0.5, 5.0).astype(np.float32)

    def score_components(self, user_id=None, additional_ratings=None, movie_id_type='tmdb'):
        """Return full-catalog predictions for fair, matched model comparisons.

        baseline, collaborative and content are each on the 0.5-5 rating scale.
        Unknown users/movies use training-only averages for the baseline.
        """
        user_id = None if user_id is None else int(user_id)
        history = dict(self.user_ratings.get(user_id, {}))
        if additional_ratings:
            for raw_mid, rating in additional_ratings.items():
                mid = self._resolve(raw_mid, movie_id_type)
                if mid not in self.id_to_col:
                    raise ValueError(f'Movie ID {raw_mid} not in catalog')
                rating = float(rating)
                if not 0.5 <= rating <= 5:
                    raise ValueError('Ratings must be between 0.5 and 5')
                history[mid] = rating
        mean = np.mean(list(history.values())) if history else 3.0
        content = np.full(len(self.movies), 3.0, dtype=np.float32)
        profile = self._profile(history)
        if profile is None:
            # Bayesian-smoothed TMDB votes, with popularity as tie-breaker.
            votes = pd.to_numeric(self.movies.get('vote_count', pd.Series(0, index=self.movies.index)), errors='coerce').fillna(0).to_numpy()
            avg = pd.to_numeric(self.movies.get('vote_average', pd.Series(0, index=self.movies.index)), errors='coerce').fillna(0).to_numpy()
            good = votes > 0
            global_mean = float(np.average(avg[good], weights=votes[good])) if np.any(good) else 6.0
            quality = (votes * avg + 100 * global_mean) / (votes + 100)
            # Scale TMDB 0-10 aggregate quality into a conservative 0.5-5 ranking score.
            content = np.clip(0.5 * quality, 0.5, 5.0).astype(np.float32)
            if not np.any(good):
                pop = pd.to_numeric(self.movies.get('popularity', pd.Series(0, index=self.movies.index)), errors='coerce').fillna(0).to_numpy()
                content += (np.log1p(pop) / max(1, np.log1p(pop).max()) * 0.1).astype(np.float32)
        if profile is not None:
            similarity = np.asarray(self.features @ profile.T).ravel()
            content = np.clip(3.0 + 2.0 * similarity, 0.5, 5.0)
        collab = np.full(len(self.movies), mean, dtype=np.float32)
        if self.svd is not None and user_id in self.user_to_row:
            idx = self.user_to_row[user_id]
            collab = np.clip(self.user_means[user_id] +
                             self.user_factors[idx] @ self.item_factors.T, 0.5, 5.0)
        # Use collaborative signal only for established training-set users and rated movies.
        n = self.user_counts.get(user_id, 0) if user_id in self.user_to_row else 0
        user_conf = n / (n + 15.0)
        item_conf = self.item_counts / (self.item_counts + 10.0)
        alpha = user_conf * item_conf if self.svd is not None else np.zeros(len(self.movies))
        hybrid = alpha * collab + (1 - alpha) * content
        # A deliberately simple, shrinkage-adjusted mean baseline. All values
        # come exclusively from training interactions, never the held-out set.
        global_mean = getattr(self, 'global_rating_mean', 3.0)
        user_mean = self.user_means.get(user_id, global_mean)
        user_n = self.user_counts.get(user_id, 0)
        adjusted_user = (user_n * user_mean + 10 * global_mean) / (user_n + 10)
        item_mean = self.item_rating_means
        item_n = self.item_counts
        adjusted_item = (item_n * item_mean + 10 * global_mean) / (item_n + 10)
        baseline = np.clip(adjusted_user + adjusted_item - global_mean, 0.5, 5.0)
        return {
            'baseline': baseline,
            'collaborative': np.asarray(collab),
            'content': np.asarray(content),
            'hybrid': np.clip(hybrid, 0.5, 5.0),
        }, history, alpha

    # Fold-in uses the fitted SVD item basis, not a retrained decomposition.
    # With few ratings, a ridge fit can be noisy; the ranker uses a conservative
    # collaborative blend and falls back to Top-K for insufficient support.
    FOLD_IN_MIN_RATINGS = 5
    FOLD_IN_REGULARIZATION = 1.0

    def fold_in_user(self, history, *, min_ratings=None, regularization=None):
        """Estimate a new/current user's SVD coordinates using ridge regression.

        R = U @ V.T was fitted to each training user's *mean-centered* ratings.
        Solve (Q.T Q + lambda I) u = Q.T (r - mean(r)), where Q consists of
        rows from the frozen SVD.components_.T corresponding to rated movies.
        Movies without an observed collaborative factor (item_counts == 0) are
        excluded. No MovieLens IDs need to be assigned to new Reelwise users.

        The ridge multiplier is relative to the average diagonal of Q.T Q:
        absolute ridge penalties are poorly scaled for sparse SVD bases.
        Returns None if there isn't enough collaborative evidence.
        """
        min_ratings = (self.FOLD_IN_MIN_RATINGS if min_ratings is None
                       else int(min_ratings))
        regularization = (self.FOLD_IN_REGULARIZATION if regularization is None
                          else float(regularization))
        if min_ratings < 1 or not np.isfinite(regularization) or regularization <= 0:
            raise ValueError('Invalid fold-in minimum ratings or regularization')
        if self.item_factors is None:
            return None
        observations = []
        for movie_id, rating in history.items():
            col = self.id_to_col.get(int(movie_id))
            if col is not None and self.item_counts[col] > 0:
                rating = float(rating)
                if not np.isfinite(rating) or not 0.5 <= rating <= 5.0:
                    raise ValueError('Ratings must be finite and between 0.5 and 5')
                observations.append((col, rating))
        if len(observations) < min_ratings:
            return None
        indices = np.array([col for col, _ in observations], dtype=np.intp)
        ratings = np.array([rating for _, rating in observations], dtype=np.float64)
        mean = float(np.mean(ratings))
        Q = np.asarray(self.item_factors[indices], dtype=np.float64)
        k = Q.shape[1]
        if k == 0 or not np.any(Q):
            return None
        gram = Q.T @ Q
        # Scale-invariant ridge: the regularizer adjusts to the norm of the
        # observed item factors. This avoids over-shrinking high-dimensional
        # catalog bases just because their per-movie factor norms are small.
        ridge_scale = float(np.trace(gram)) / k
        if ridge_scale <= 0 or not np.isfinite(ridge_scale):
            return None
        lhs = gram + (regularization * ridge_scale) * np.eye(k)
        rhs = Q.T @ (ratings - mean)
        factors = np.linalg.solve(lhs, rhs)
        if not np.all(np.isfinite(factors)):
            return None
        return {'factors': factors, 'mean': mean, 'support': len(observations)}

    @staticmethod
    def _fold_in_weight(support):
        """Conservative starting weights; tune via cold-start holdouts."""
        if support < 5:
            return 0.0
        if support < 10:
            return 0.25
        if support < 20:
            return 0.50
        return 0.90

    # Production *ranking* configuration. These are experiment-selected starting
    # points, not calibrated star-rating predictors. Tests/evaluate.py continue
    # to use the original score_components() without modification.
    TOP_K = 10
    NEW_USER_POPULARITY_WEIGHT = 0.50
    ESTABLISHED_COLLABORATIVE_WEIGHT = 0.90
    # Explicit onboarding choices are a weak prior, not a genre filter.
    # As observed ratings grow, implicit preferences should take precedence.
    GENRE_WEIGHT = 0.08
    GENRE_DECAY_RATINGS = 5.0
    MAX_PREFERRED_GENRE_FRACTION = 0.60

    def genre_preference_weight(self, rating_count):
        """Smoothly diminish the influence of stated genres with evidence."""
        return self.GENRE_WEIGHT / (1.0 + max(0, rating_count) / self.GENRE_DECAY_RATINGS)

    def _ranking_scores(self, user_id=None, additional_ratings=None,
                        movie_id_type='tmdb', use_fold_in=False):
        """Frontend ranking: trained users 90/10, newcomers Top-K/popularity.

        For a trained user, movies with zero collaborative interactions use
        the content score alone. Unknown users never use another user's SVD
        representation. Genre weighting is applied afterwards.
        """
        components, history, _ = self.score_components(
            user_id, additional_ratings, movie_id_type
        )
        positive = any(float(r) >= 4.0 and int(mid) in self.id_to_col
                       for mid, r in history.items())
        if positive:
            content_rank = self.score_topk_content(
                history, top_k=self.TOP_K,
                popularity_weight=self.NEW_USER_POPULARITY_WEIGHT
            )
        else:
            # Exactly the same global popularity ordering for all users with
            # no positive evidence, independent of the chosen Top-K weight.
            counts = self.item_counts.astype(np.float64)
            peak = float(np.log1p(counts.max())) if counts.size else 0.0
            prior = np.log1p(counts) / peak if peak > 0 else np.zeros(len(counts))
            content_rank = (0.5 + 4.5 * prior).astype(np.float32)

        trained = (self.svd is not None and user_id is not None
                   and int(user_id) in self.user_to_row)
        alpha = np.zeros(len(self.movies), dtype=np.float32)
        if trained:
            alpha[self.item_counts > 0] = self.ESTABLISHED_COLLABORATIVE_WEIGHT
            scores = (alpha * components['collaborative']
                      + (1.0 - alpha) * content_rank)
            strategy = 'collaborative_topk_hybrid'
        elif use_fold_in:
            folded = self.fold_in_user(history)
            if folded is not None:
                weight = self._fold_in_weight(folded['support'])
                alpha[self.item_counts > 0] = weight
                collaborative = np.clip(
                    folded['mean'] + self.item_factors @ folded['factors'],
                    0.5, 5.0
                )
                scores = alpha * collaborative + (1.0 - alpha) * content_rank
                strategy = 'fold_in_topk_hybrid'
            else:
                scores = content_rank
                strategy = 'topk_popularity' if positive else 'popularity'
        else:
            scores = content_rank
            strategy = 'topk_popularity' if positive else 'popularity'
        return np.asarray(scores, dtype=np.float32), history, alpha, strategy

    def _scores(self, user_id=None, additional_ratings=None, movie_id_type='tmdb'):
        """Preserve existing API behavior for the FastAPI application."""
        components, history, alpha = self.score_components(
            user_id, additional_ratings, movie_id_type
        )
        return components['hybrid'], history, alpha

    def _genre_matches(self, preferred_genres):
        """Movie-level matching of explicitly selected genres, 0..1.

        Matches canonical TMDB genres, allowing the common Sci-Fi alias.
        Only called at inference time; no retraining necessary.
        """
        aliases = {'sci-fi': 'science fiction'}
        selected = {
            aliases.get(str(g).strip().casefold(), str(g).strip().casefold())
            for g in (preferred_genres or []) if str(g).strip()
        }
        scores = np.zeros(len(self.movies), dtype=np.float32)
        if not selected:
            return scores
        for i, genres in enumerate(self.genre):
            tokens = {
                aliases.get(g.strip().casefold(), g.strip().casefold())
                for g in str(genres).split(',') if g.strip()
            }
            if tokens:
                scores[i] = len(tokens & selected) / len(tokens)
        return scores

    def _add_genre_preferences(self, scores, preferred_genres, genre_weight=None):
        """Blend genre affinity with a 0.5..5 existing hybrid score."""
        matches = self._genre_matches(preferred_genres)
        if genre_weight is None:
            genre_weight = self.GENRE_WEIGHT
        if not preferred_genres or genre_weight <= 0:
            return scores, matches
        # Both components share the same [0,1] range before blending.
        blended = (1 - genre_weight) * ((scores - 0.5) / 4.5) + genre_weight * matches
        return np.clip(0.5 + 4.5 * blended, 0.5, 5.0), matches

    def predict(self, user_id, movie_id, movie_id_type='tmdb', additional_ratings=None, preferred_genres=None, mode='hybrid'):
        mid = self._resolve(movie_id, movie_id_type)
        if mid not in self.id_to_col:
            raise ValueError('Movie ID not found in catalog')
        components, _, _ = self.score_components(user_id, additional_ratings, movie_id_type)
        if mode not in components:
            raise ValueError(f'Unknown prediction mode: {mode}')
        scores = components[mode]
        if mode == 'hybrid':
            scores, _ = self._add_genre_preferences(scores, preferred_genres)
        return float(scores[self.id_to_col[mid]])

    def _diversified_top_indices(self, scores, genre_matches, k, enforce_cap):
        """Rank by model score, capping selected-genre saturation when possible.

        This is a presentation-time diversity guard, NOT a genre exclusion:
        recommendations outside the preferred genres must still score well.
        With insufficient other-genre candidates, backfill the remaining slots.
        """
        available = np.flatnonzero(np.isfinite(scores))
        if not enforce_cap:
            chosen = available[np.argpartition(-scores[available], k - 1)[:k]]
            return chosen[np.argsort(-scores[chosen], kind='stable')]

        def top_in_group(indices):
            if not len(indices):
                return np.empty(0, dtype=np.int64)
            size = min(k, len(indices))
            best = indices[np.argpartition(-scores[indices], size - 1)[:size]]
            return best[np.argsort(-scores[best], kind='stable')]

        preferred = top_in_group(available[genre_matches[available] > 0])
        other = top_in_group(available[genre_matches[available] <= 0])
        cap = max(1, int(np.ceil(k * self.MAX_PREFERRED_GENRE_FRACTION)))
        selected = []
        pi = oi = preferred_count = 0
        while len(selected) < k:
            can_prefer = pi < len(preferred) and (preferred_count < cap or oi >= len(other))
            can_other = oi < len(other)
            if can_prefer and (not can_other or scores[preferred[pi]] >= scores[other[oi]]):
                selected.append(preferred[pi])
                pi += 1
                preferred_count += 1
            elif can_other:
                selected.append(other[oi])
                oi += 1
            elif pi < len(preferred):
                selected.append(preferred[pi])
                pi += 1
            else:
                break
        return np.asarray(selected, dtype=np.int64)

    def recommend(self, user_id=None, additional_ratings=None, top_n=10,
                  movie_id_type='tmdb', preferred_genres=None, use_fold_in=False):
        if top_n < 1:
            return []
        scores, history, alpha, strategy = self._ranking_scores(
            user_id, additional_ratings, movie_id_type, use_fold_in=use_fold_in
        )
        scores, genre_matches = self._add_genre_preferences(
            scores, preferred_genres,
            genre_weight=self.genre_preference_weight(len(history))
        )
        for mid in history:
            if mid in self.id_to_col:
                scores[self.id_to_col[mid]] = -np.inf
        k = min(top_n, int(np.isfinite(scores).sum()))
        if k == 0:
            return []
        order = self._diversified_top_indices(
            scores, genre_matches, k,
            enforce_cap=bool(preferred_genres),
        )
        result = []
        for i in order:
            movie = self.movies.iloc[int(i)]
            result.append({
                'tmdb_id': int(movie.id),
                'title': str(movie.title),
                'genres': str(movie.get('genres', '')),
                'score': round(float(scores[i]), 3),
                'collaborative_weight': round(float(alpha[i]), 3),
                'genre_match': round(float(genre_matches[i]), 3),
                'strategy': strategy,
            })
        return result
