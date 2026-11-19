> **AWS EC2 deployment:** See [DEPLOY_EC2.md](DEPLOY_EC2.md) for secure cookies,
> rate limiting, TLS, Nginx, systemd, backups, and production setup.

# Reelwise — React frontend + FastAPI recommendation backend

This project merges the supplied Reelwise React/Vite design with the latest hybrid movie recommendation engine. It uses TMDB metadata + MovieLens ratings and keeps training separate from API serving.

## Prerequisites
- Python 3.10+ with the packages from `backend/requirements.txt`
- Node.js 22+ and npm (or pnpm)
- A **locally trained, trusted** `backend/models/hybrid.joblib` file. **The model and the large datasets are not included in the ZIP.**

## 1. Prepare the model (if you don't already have it)

Place `TMDB_movie_dataset_v11.csv`, MovieLens `ratings.csv`, and `links.csv` under `backend/data/raw/`.

```bash
cd backend
python -m pip install -r requirements.txt
python filter_catalog.py --size 150000 --min-ratings 5
python train.py --skip-evaluation
```

If you already trained the prior project, you can instead copy your `hybrid.joblib` and processed data into the corresponding folders (do not load untrusted model files). A compatible model is needed before searching or requesting recommendations.

## 2. Start FastAPI

```bash
cd backend
uvicorn backend:app --reload --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health` (`model_ready: true`); Swagger UI: `http://127.0.0.1:8000/docs`.

SQLite will be created at `backend/data/app.sqlite3`. Old SQLite databases automatically gain the password-hash column and watchlist table. **Legacy accounts created using `/users` don't have passwords**; they can still use their original bearer token but must register a new password-based account to use the React login screen. No password migration or account recovery has been built.

## 3. Start React

```bash
cd frontend
npm install
cp .env.example .env.local   # Windows: copy .env.example .env.local
npm run dev
```

Open `http://localhost:8443` on the development PC, or `http://YOUR-LAN-IP:8443` from another device on the same local network. **The frontend now uses `VITE_API_URL=/api`**, so browser API calls go through Vite's proxy to `http://127.0.0.1:8000` on the development machine. There is no hardcoded LAN IP or CORS change required. Keep FastAPI bound to `127.0.0.1` and expose only Vite on your trusted LAN. Restart Vite after changing `.env.local`.

**Connectivity check from your phone:** open `http://YOUR-LAN-IP:8443/api/health`; you should see JSON containing `model_ready`. If not, verify Vite and FastAPI are running and Windows Firewall allows inbound port **8443** on your private network. The proxy is for Vite dev mode; production hosting requires an equivalent reverse proxy for `/api`. Do not expose the development server to the public internet.


## 4. Online collaborative fold-in (no retraining after ratings)

**Fold-in is enabled for Reelwise API recommendations.** On every `/recommendations`
request the backend uses the user's current SQLite ratings and the trained, frozen
SVD movie factors. If at least **5** rated movies have MovieLens collaborative
factors, it solves a small regularized least-squares system to estimate the user's
latent vector in memory. It does **not** re-fit SVD, write a new `.joblib`, or
require the original 25-million-rating dataset to be on the server.

- **0–4 linked ratings:** original Top-K TF-IDF + popularity ranking.
- **5–9 linked ratings:** 25% fold-in collaborative / 75% Top-K + popularity.
- **10–19 linked ratings:** 50% fold-in collaborative / 50% Top-K + popularity.
- **20+ linked ratings:** 90% fold-in collaborative / 10% Top-K + popularity.
- **Users with a current trained SVD factor:** original 90/10 hybrid remains in use.
- **Movies with no collaborative interactions:** content/popularity fallback only.
- **TMDB-only ratings:** influence Top-K immediately but do not count toward the
  minimum of five linked collaborative observations.

These fold-in confidence weights and the default ridge multiplier (1.0, normalized
by the item-factor Gram-matrix scale) are **conservative starting choices**, not
validated optimal values. Evaluate cold-start NDCG@10 before changing or claiming
an improvement in ranking quality. `score_components()` and the standalone
holdout evaluator retain their earlier definitions for comparison.

### Optional offline full retraining

The existing `batch_retrain.py` remains available if you later want to refresh
the **movie** latent factors from new population-wide interactions or updated
catalog data. **Do not enable the daemon or EC2 timer just to incorporate a new
user's ratings.** Fold-in already handles them on demand. A full retrain is still
useful occasionally, but consumes much more CPU/RAM and requires the large
MovieLens training CSVs. Example manual command:

```bash
cd backend
python batch_retrain.py --once
```

If you retrain, the script creates a temporary model, atomically replaces the
trusted trained model after success, and FastAPI reloads it when the file changes.

## Features connected

- Register with username/password; sign in again with saved credentials. Passwords are PBKDF2-HMAC-SHA256 hashed with a random salt; the browser uses HttpOnly cookie sessions with CSRF protection. Bearer tokens remain available for test/client compatibility.
- Genre onboarding/preferences (`GET /genres`, `GET/PUT /users/me/genres`).
- Search the trained TMDB catalog and open movie details with plot summary/poster.
- Save/update a 1–5-star rating (`PUT /ratings`), see rated movies under Watched; saving a rating removes a movie from Want to watch.
- Persist Want to watch via `GET/POST/DELETE /watchlist`.
- Get up to 100 ranked recommendations, show 10 at a time, and browse next results with Reroll. This **pages the ranked list**; it doesn't refit or randomize the trained model. Random movie selects from the current 100 recommendation candidates.
- Live ranking uses Top-K (K=10), a 50/50 similarity–popularity content score, user-factor fold-in after five linked ratings, 90/10 for existing trained users, and explicit genre adjustment.

## API additions made for Reelwise

| Method | Path | Description |
| --- | --- | --- |
| POST | `/auth/register` | `{username,password}` -> account + token |
| POST | `/auth/login` | Verify password, issue new token |
| POST | `/auth/logout` | Revoke bearer token |
| GET | `/watchlist` | Current user's saved movie details |
| POST | `/watchlist` | `{tmdb_id}`: save a movie |
| DELETE | `/watchlist/{tmdb_id}` | Remove a saved movie |

The original `/users` endpoint is kept for backward compatibility with old scripts/tests. The existing model APIs remain unchanged.

## Backend tests

```bash
cd backend
python -m pytest -q
```

The backend's test suite uses a synthetic fitted recommender and a temporary SQLite DB; it does **not** require the full trained model.

## Current limitations

- The provided frontend originally had **mock friends' activity and profiles**, so their navigation has been hidden rather than presenting fictional data as real. Social APIs have not been implemented.
- Watched is defined as **rated** for now. There is no separate unrated-watched status.
- Forgot-password and recovery are not available. Use HTTPS, the included production protections, and controlled test-user invitations; this is not a security-audited public service.
- Vite's original Figma-specific config remains in place to preserve the supplied layout.
- Movie catalog and model must be present for catalog/search/recommendation endpoints. User register/login and genre choices can work without a fitted model.
- Existing/legacy bearer-token accounts have no password; new users should use `/auth/register`.
- Images are requested from TMDB CDN; ensure poster use complies with TMDB attribution/terms.
- Current app ratings update fold-in collaborative predictions **immediately**, with no full retraining. Optional batch retraining refreshes the frozen movie factors.
