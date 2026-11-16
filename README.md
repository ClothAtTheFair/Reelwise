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


## 4. Automatic periodic collaborative retraining

Ratings continue to personalize Top-K recommendations **immediately**. The SVD model is refreshed in the background by a **separate retraining process**, never inside an API request.

From a **third terminal**, run this on the development machine:

```bash
cd backend
python batch_retrain.py --daemon --interval-hours 24
```

The process checks immediately on start and every 24 hours while it is running. It trains only when saved app ratings have changed since its last successful run, and skips when there are no app ratings. **Keep this terminal/process running** for recurring retraining. On the first run with existing app ratings, it trains immediately; with ~25 million MovieLens ratings, this can take substantial time and memory. Stop with Ctrl+C (after an in-progress training process finishes).

**Recommended for Windows when you don't want a terminal left open:** use Windows Task Scheduler to create a daily task. Set *Program/script* to the absolute path of your Python executable (run `python -c "import sys; print(sys.executable)"`), *Add arguments* to the absolute path of `backend/batch_retrain.py` followed by `--once`, and *Start in* to the absolute `backend` directory. Alternatively, from a terminal manually run:

```bash
cd backend
python batch_retrain.py --once
```

Optional: `--force` performs training even when the app-rating snapshot has not changed (e.g. following a manual change of base data). `--factors 40` and `--seed 42` preserve the original training settings; `--app-db`, `--data`, and `--model` can override paths. For periodic runs, supply the **same** database and model paths FastAPI uses.

The retrainer writes a temporary `.joblib` in the models directory. **Only after successful training** does it atomically replace `models/hybrid.joblib`; on failure the existing model remains usable. FastAPI notices the changed file and reloads on its next model request (a long-running request may still use its previous in-memory model). The retrainer records successful progress in `models/retrain_state.json` and prevents overlapping runs with `models/hybrid.joblib.retrain.lock`. If the process crashes and leaves that lock behind, confirm no retraining process is running before removing it manually. Keep the model directory writable; train/load only trusted model files.

**Note:** App ratings on TMDB-only movies affect Top-K immediately but cannot be mapped into the current MovieLens-ID SVD training. The batch pipeline still snapshots all app ratings so its staleness detection is accurate.

## Features connected

- Register with username/password; sign in again with saved credentials. Passwords are PBKDF2-HMAC-SHA256 hashed with a random salt; bearer tokens are stored in the browser **sessionStorage** (prototype only) and rotated on login/logout.
- Genre onboarding/preferences (`GET /genres`, `GET/PUT /users/me/genres`).
- Search the trained TMDB catalog and open movie details with plot summary/poster.
- Save/update a 1–5-star rating (`PUT /ratings`), see rated movies under Watched; saving a rating removes a movie from Want to watch.
- Persist Want to watch via `GET/POST/DELETE /watchlist`.
- Get up to 100 ranked recommendations, show 10 at a time, and browse next results with Reroll. This **pages the ranked list**; it doesn't refit or randomize the trained model. Random movie selects from the current 100 recommendation candidates.
- Live recommendation scoring retains the backend's Top-K (K=10), 50/50 similarity–popularity new-user ranking, established-user 90/10 collaborative/content ranking, and explicit genre adjustment.

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
- Forgot-password and recovery are not available. Don't use this prototype authentication for a public launch without additional security controls (rate limits, HTTPS, recovery, more robust session management).
- Vite's original Figma-specific config remains in place to preserve the supplied layout.
- Movie catalog and model must be present for catalog/search/recommendation endpoints. User register/login and genre choices can work without a fitted model.
- Existing/legacy bearer-token accounts have no password; new users should use `/auth/register`.
- Images are requested from TMDB CDN; ensure poster use complies with TMDB attribution/terms.
- Collaborative factors for app users update after running `batch_retrain.py` (daily daemon or OS-scheduled `--once`). Content-based ratings take effect immediately. FastAPI automatically reloads the replaced model on its next request.
