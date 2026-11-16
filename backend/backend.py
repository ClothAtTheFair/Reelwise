"""FastAPI application for the hybrid recommender.

Run: uvicorn backend:app --reload
The trained model is read-only during serving. Ratings are persisted in SQLite.
"""
import hashlib
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from config import production, validate_production, integer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv('MODEL_PATH', str(ROOT / 'models/hybrid.joblib')))
DB_PATH = Path(os.getenv('APP_DB_PATH', str(ROOT / 'data/app.sqlite3')))
APP_USER_OFFSET = 1_000_000_000  # Different namespace from MovieLens IDs

@asynccontextmanager
async def lifespan(application):
    validate_production()
    init_db()
    yield


app = FastAPI(title='Hybrid Movie Recommender API', version='2.1.0', lifespan=lifespan,
              docs_url=None if production() else '/docs',
              redoc_url=None if production() else '/redoc',
              openapi_url=None if production() else '/openapi.json', debug=False)
# Configure CORS_ORIGINS as a comma-separated list of your frontend URLs.
origins = ([] if production() else [x.strip() for x in os.getenv('CORS_ORIGINS', 'http://localhost:8443,http://127.0.0.1:8443,http://localhost:5173,http://localhost:3000').split(',') if x.strip()])
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET', 'POST', 'PUT', 'DELETE'], allow_headers=['Authorization', 'Content-Type', 'X-CSRF-Token'])

SESSION_COOKIE = 'reelwise_session'
CSRF_COOKIE = 'reelwise_csrf'
UNSAFE_METHODS = {'POST', 'PUT', 'PATCH', 'DELETE'}
_auth_attempts = defaultdict(deque)
_auth_limit_lock = threading.Lock()


def check_auth_rate_limit(request: Request, username: str = '') -> None:
    """Simple single-worker limiter for EC2. Bind Uvicorn to one worker."""
    peer = request.client.host if request.client else 'unknown'
    keys = [(peer, '*'), (peer, username.casefold())]
    now = time.monotonic()
    window = integer('AUTH_RATE_LIMIT_WINDOW_SECONDS', 900)
    limit = integer('AUTH_RATE_LIMIT_ATTEMPTS', 8)
    with _auth_limit_lock:
        for key in keys:
            entries = _auth_attempts[key]
            while entries and entries[0] < now - window:
                entries.popleft()
            max_attempts = limit * 4 if key[1] == '*' else limit
            if len(entries) >= max_attempts:
                raise HTTPException(429, detail='Too many authentication attempts. Try again later.',
                                    headers={'Retry-After': str(window)})
        for key in keys:
            _auth_attempts[key].append(now)
        if len(_auth_attempts) > 10000:
            for old_key in list(_auth_attempts):
                old = _auth_attempts[old_key]
                if not old or old[-1] < now - window:
                    del _auth_attempts[old_key]


@app.middleware('http')
async def security_headers(request: Request, call_next):
    if production() and request.method in UNSAFE_METHODS:
        expected_origin = os.environ.get('FRONTEND_ORIGIN', '').rstrip('/')
        if request.headers.get('origin') != expected_origin:
            return Response('Origin check failed', status_code=403)
    if production() and request.url.path.startswith('/api/'):
        # /api prefix should be removed by Nginx; prevents accidental misrouting.
        return Response(status_code=404)
    length = request.headers.get('content-length')
    if length and request.method in UNSAFE_METHODS:
        try:
            if int(length) > 16384:
                return Response('Request too large', status_code=413)
        except ValueError:
            return Response('Invalid Content-Length', status_code=400)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['X-Frame-Options'] = 'DENY'
    if request.url.path.startswith(('/auth/', '/users/me', '/ratings', '/watchlist', '/recommendations')):
        response.headers['Cache-Control'] = 'no-store'
    return response

_model = None
_model_mtime = None
_model_lock = threading.RLock()


def connection():
    db_path = Path(os.getenv('APP_DB_PATH', str(DB_PATH)))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=15000')
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with next_connection() as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                password_hash TEXT,
                token_expires_at TEXT,
                csrf_hash TEXT
            );
            CREATE TABLE IF NOT EXISTS ratings (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                tmdb_id INTEGER NOT NULL,
                rating REAL NOT NULL CHECK(rating >= 0.5 AND rating <= 5.0),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, tmdb_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(tmdb_id);
            CREATE TABLE IF NOT EXISTS watchlist (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                tmdb_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, tmdb_id)
            );
            CREATE TABLE IF NOT EXISTS user_genre_preferences (
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                genre TEXT NOT NULL,
                preference_weight REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY(user_id, genre)
            );
        ''')
        # Upgrade databases created before password login existed.
        columns = {row['name'] for row in db.execute('PRAGMA table_info(users)')}
        if 'password_hash' not in columns:
            db.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
        if 'token_expires_at' not in columns:
            db.execute('ALTER TABLE users ADD COLUMN token_expires_at TEXT')
        if 'csrf_hash' not in columns:
            db.execute('ALTER TABLE users ADD COLUMN csrf_hash TEXT')
        db.commit()


@contextmanager
def next_connection():
    generator = connection()
    conn = next(generator)
    try:
        yield conn
    finally:
        generator.close()


def get_model():
    global _model, _model_mtime
    path = Path(os.getenv('MODEL_PATH', str(MODEL_PATH)))
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail='Model not trained. Run filter_catalog.py and train.py first.')
    with _model_lock:
        if _model is None or _model_mtime != (str(path), mtime):
            try:
                _model = joblib.load(path)  # Only load YOUR OWN trusted model files.
                _model_mtime = (str(path), mtime)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f'Model cannot be loaded: {type(exc).__name__}') from exc
        return _model


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def authenticated_user(request: Request, authorization: Annotated[str | None, Header()] = None) -> int:
    # Browser clients authenticate using an HttpOnly cookie. Old bearer tokens remain
    # compatible in development/tests only, to support legacy scripts and tests.
    bearer = (authorization[7:].strip() if authorization and authorization.startswith('Bearer ') else None)
    if production() and bearer:
        raise HTTPException(status_code=401, detail='Cookie authentication required')
    token = (bearer or request.cookies.get(SESSION_COOKIE)) if not production() else request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail='Authentication required')
    with next_connection() as db:
        row = db.execute('SELECT id, token_expires_at, csrf_hash FROM users WHERE token_hash=?',
                         (token_digest(token),)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail='Invalid session')
    expires = row['token_expires_at']
    if expires and datetime.fromisoformat(expires) <= datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail='Session expired')
    if production() and not expires:
        raise HTTPException(status_code=401, detail='Legacy session must log in again')
    if request.cookies.get(SESSION_COOKIE) and request.method in UNSAFE_METHODS:
        supplied = request.headers.get('X-CSRF-Token', '')
        expected = row['csrf_hash']
        if not expected or not supplied or not secrets.compare_digest(token_digest(supplied), expected):
            raise HTTPException(status_code=403, detail='CSRF check failed')
        origin = request.headers.get('origin')
        allowed_origin = os.getenv('FRONTEND_ORIGIN', '').rstrip('/')
        if production() and origin != allowed_origin:
            raise HTTPException(status_code=403, detail='Origin check failed')
    return int(row['id'])


def issue_session(response: Response, user_id: int, username: str, db: sqlite3.Connection) -> dict:
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(32)
    ttl_hours = integer('AUTH_TOKEN_TTL_HOURS', 24)
    expires = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    db.execute('UPDATE users SET token_hash=?, csrf_hash=?, token_expires_at=? WHERE id=?',
               (token_digest(token), token_digest(csrf), expires.isoformat(), user_id))
    if os.getenv('APP_ENV') != 'test':
        response.set_cookie(SESSION_COOKIE, token, httponly=True, secure=production(),
                            samesite='lax', path='/', max_age=ttl_hours * 3600)
        response.set_cookie(CSRF_COOKIE, csrf, httponly=False, secure=production(),
                            samesite='lax', path='/', max_age=ttl_hours * 3600)
    # Legacy bearer access for local scripts, never exposed by public production API.
    result = {'user_id': user_id, 'username': username}
    if not production():
        result['token'] = token
    return result


def expire_session_cookies(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path='/', secure=production(), samesite='lax')
    response.delete_cookie(CSRF_COOKIE, path='/', secure=production(), samesite='lax')


class NewUser(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r'^[a-zA-Z0-9_-]+$')


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r'^[a-zA-Z0-9_-]+$')
    password: str = Field(min_length=8, max_length=128)


def make_password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 310_000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        algorithm, iterations, salt, expected = stored.split('$')
        if algorithm != 'pbkdf2_sha256':
            return False
        result = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), int(iterations))
        return secrets.compare_digest(result, bytes.fromhex(expected))
    except (ValueError, TypeError):
        return False


@app.post('/auth/register', status_code=201)
def register(data: Credentials, request: Request, response: Response):
    check_auth_rate_limit(request, data.username)
    if len(data.password) < 12 and production():
        raise HTTPException(422, detail='Password must have at least 12 characters')
    with next_connection() as db:
        try:
            cursor = db.execute(
                'INSERT INTO users(username, token_hash, password_hash) VALUES (?,?,?)',
                (data.username, token_digest(secrets.token_urlsafe(32)), make_password_hash(data.password))
            )
            result = issue_session(response, cursor.lastrowid, data.username, db)
            db.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(409, detail='Username unavailable')
    return result


@app.post('/auth/login')
def login(data: Credentials, request: Request, response: Response):
    check_auth_rate_limit(request, data.username)
    with next_connection() as db:
        row = db.execute('SELECT id, username, password_hash FROM users WHERE username=?', (data.username,)).fetchone()
        if row is None or not check_password(data.password, row['password_hash']):
            raise HTTPException(401, detail='Invalid username or password')
        result = issue_session(response, int(row['id']), str(row['username']), db)
        db.commit()
    return result


@app.post('/auth/logout', status_code=204)
def logout(response: Response, user_id: int = Depends(authenticated_user)):
    with next_connection() as db:
        db.execute('UPDATE users SET token_hash=?, csrf_hash=NULL, token_expires_at=NULL WHERE id=?',
                   (token_digest(secrets.token_urlsafe(32)), user_id))
        db.commit()
    expire_session_cookies(response)


@app.get('/auth/session')
def session(user_id: int = Depends(authenticated_user)):
    with next_connection() as db:
        row = db.execute('SELECT username FROM users WHERE id=?', (user_id,)).fetchone()
    return {'user_id': user_id, 'username': row['username']}


@app.delete('/users/me', status_code=204)
def delete_account(response: Response, user_id: int = Depends(authenticated_user)):
    with next_connection() as db:
        db.execute('DELETE FROM users WHERE id=?', (user_id,))
        db.commit()
    expire_session_cookies(response)


class RatingInput(BaseModel):
    tmdb_id: int = Field(gt=0)
    rating: float = Field(ge=0.5, le=5.0)

    @field_validator('rating')
    @classmethod
    def half_star_increments(cls, v):
        if abs(v * 2 - round(v * 2)) > 1e-8:
            raise ValueError('Use half-star increments (0.5, 1.0, ..., 5.0)')
        return v


VALID_GENRES = (
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
    'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
)
GENRE_LOOKUP = {value.casefold(): value for value in VALID_GENRES}
GENRE_LOOKUP['sci-fi'] = 'Science Fiction'


class GenrePreferencesRequest(BaseModel):
    genres: list[str] = Field(default_factory=list, max_length=30)


class GenrePreferencesResponse(BaseModel):
    genres: list[str]


def user_genres(user_id: int) -> list[str]:
    with next_connection() as db:
        rows = db.execute(
            'SELECT genre FROM user_genre_preferences WHERE user_id=? ORDER BY genre',
            (user_id,)
        ).fetchall()
    return [row['genre'] for row in rows]


@app.get('/genres')
def available_genres():
    return {'genres': list(VALID_GENRES)}


@app.get('/users/me/genres', response_model=GenrePreferencesResponse)
def get_genre_preferences(user_id: int = Depends(authenticated_user)):
    return {'genres': user_genres(user_id)}


@app.put('/users/me/genres', response_model=GenrePreferencesResponse)
def put_genre_preferences(payload: GenrePreferencesRequest, user_id: int = Depends(authenticated_user)):
    genres = set()
    for raw_genre in payload.genres:
        genre = GENRE_LOOKUP.get(raw_genre.strip().casefold())
        if genre is None:
            raise HTTPException(status_code=422, detail=f'Invalid genre: {raw_genre}')
        genres.add(genre)
    # Single transaction: invalid input cannot erase existing preferences.
    with next_connection() as db:
        db.execute('DELETE FROM user_genre_preferences WHERE user_id=?', (user_id,))
        db.executemany(
            'INSERT INTO user_genre_preferences(user_id, genre) VALUES (?, ?)',
            [(user_id, genre) for genre in sorted(genres)]
        )
        db.commit()
    return {'genres': sorted(genres)}


def user_ratings(user_id: int) -> dict[int, float]:
    with next_connection() as db:
        rows = db.execute('SELECT tmdb_id,rating FROM ratings WHERE user_id=?', (user_id,)).fetchall()
    return {int(row['tmdb_id']): float(row['rating']) for row in rows}


def movie_details(model, i: int):
    row = model.movies.iloc[i]
    def val(name):
        x = row.get(name)
        return None if x is None or pd.isna(x) else x
    return {
        'tmdb_id': int(row['id']),
        'title': str(row['title']),
        'genres': str(val('genres') or ''),
        'overview': str(val('overview') or ''),
        'release_date': str(val('release_date') or ''),
        'poster_path': str(val('poster_path') or ''),
        'vote_average': float(val('vote_average')) if val('vote_average') is not None else None,
        'vote_count': int(val('vote_count')) if val('vote_count') is not None else None,
    }


@app.get('/health')
def health():
    path = Path(os.getenv('MODEL_PATH', str(MODEL_PATH)))
    return {'status': 'ok', 'model_ready': path.is_file()}


@app.post('/users', status_code=status.HTTP_201_CREATED)
def create_user(data: NewUser):
    if production():
        raise HTTPException(status_code=404, detail='Not found')
    token = secrets.token_urlsafe(32)
    with next_connection() as db:
        try:
            cursor = db.execute('INSERT INTO users(username,token_hash) VALUES (?,?)', (data.username, token_digest(token)))
            db.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail='Username already exists')
    return {'user_id': cursor.lastrowid, 'username': data.username, 'token': token}


@app.get('/users/me')
def me(user_id: int = Depends(authenticated_user)):
    with next_connection() as db:
        row = db.execute('SELECT username,created_at FROM users WHERE id=?', (user_id,)).fetchone()
    return {'user_id': user_id, 'username': row['username'], 'created_at': row['created_at'], 'rating_count': len(user_ratings(user_id))}


@app.get('/movies')
def search_movies(q: str = Query(min_length=1, max_length=120), limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)):
    model = get_model()
    # Literal substring match (not regex); case-insensitive.
    mask = model.movies['title'].fillna('').astype(str).str.contains(q, case=False, regex=False)
    positions = mask.to_numpy().nonzero()[0]
    return {'total': len(positions), 'items': [movie_details(model, int(i)) for i in positions[offset:offset + limit]]}


@app.get('/movies/{tmdb_id}')
def get_movie(tmdb_id: int):
    model = get_model()
    if tmdb_id not in model.id_to_col:
        raise HTTPException(status_code=404, detail='Movie not found')
    return movie_details(model, model.id_to_col[tmdb_id])


@app.get('/ratings')
def list_ratings(user_id: int = Depends(authenticated_user)):
    return [{'tmdb_id': mid, 'rating': score} for mid, score in sorted(user_ratings(user_id).items())]


@app.put('/ratings')
def put_rating(data: RatingInput, user_id: int = Depends(authenticated_user)):
    model = get_model()
    if data.tmdb_id not in model.id_to_col:
        raise HTTPException(status_code=404, detail='Movie not found in training catalog')
    with next_connection() as db:
        db.execute('''INSERT INTO ratings(user_id,tmdb_id,rating) VALUES (?,?,?)
            ON CONFLICT(user_id,tmdb_id) DO UPDATE SET rating=excluded.rating, updated_at=CURRENT_TIMESTAMP''',
            (user_id, data.tmdb_id, data.rating))
        db.execute('DELETE FROM watchlist WHERE user_id=? AND tmdb_id=?', (user_id, data.tmdb_id))
        db.commit()
    return {'tmdb_id': data.tmdb_id, 'rating': data.rating, 'saved': True}


@app.delete('/ratings/{tmdb_id}', status_code=204)
def delete_rating(tmdb_id: int, user_id: int = Depends(authenticated_user)):
    with next_connection() as db:
        db.execute('DELETE FROM ratings WHERE user_id=? AND tmdb_id=?', (user_id, tmdb_id))
        db.commit()


class WatchlistInput(BaseModel):
    tmdb_id: int = Field(gt=0)


@app.get('/watchlist')
def get_watchlist(user_id: int = Depends(authenticated_user)):
    model = get_model()
    with next_connection() as db:
        rows = db.execute('SELECT tmdb_id FROM watchlist WHERE user_id=? ORDER BY created_at DESC, rowid DESC', (user_id,)).fetchall()
    return {'items': [movie_details(model, model.id_to_col[int(r['tmdb_id'])])
                      for r in rows if int(r['tmdb_id']) in model.id_to_col]}


@app.post('/watchlist', status_code=201)
def add_watchlist(data: WatchlistInput, user_id: int = Depends(authenticated_user)):
    model = get_model()
    if data.tmdb_id not in model.id_to_col:
        raise HTTPException(404, detail='Movie not found')
    with next_connection() as db:
        db.execute('INSERT OR IGNORE INTO watchlist(user_id,tmdb_id) VALUES (?,?)', (user_id, data.tmdb_id))
        db.commit()
    return {'tmdb_id': data.tmdb_id, 'saved': True}


@app.delete('/watchlist/{tmdb_id}', status_code=204)
def remove_watchlist(tmdb_id: int, user_id: int = Depends(authenticated_user)):
    with next_connection() as db:
        db.execute('DELETE FROM watchlist WHERE user_id=? AND tmdb_id=?', (user_id, tmdb_id))
        db.commit()


@app.get('/recommendations')
def recommendations(top_n: int = Query(10, ge=1, le=100), user_id: int = Depends(authenticated_user)):
    model = get_model()
    ratings = user_ratings(user_id)
    genres = user_genres(user_id)
    # Periodically refitted application users are assigned stable synthetic IDs.
    # Don't use a stale SVD factor when their saved ratings have changed.
    internal_uid = APP_USER_OFFSET + user_id
    trained_history = model.user_ratings.get(internal_uid)
    snapshot = getattr(model, 'app_rating_snapshots', {}).get(internal_uid, trained_history)
    training_uid = (internal_uid if trained_history is not None and snapshot == ratings else None)
    try:
        results = model.recommend(
            user_id=training_uid, additional_ratings=ratings, top_n=top_n,
            preferred_genres=genres
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    strategy = (results[0]['strategy'] if results else
                ('collaborative_topk_hybrid' if training_uid is not None else
                 'topk_popularity' if any(r >= 4 for r in ratings.values()) else 'popularity'))
    return {
        'user_id': user_id,
        'rating_count': len(ratings),
        'positive_rating_count': sum(r >= 4 for r in ratings.values()),
        'personalization': strategy + ('+genres' if genres else ''),
        'recommendation_strategy': strategy,
        'preferred_genres': genres,
        'scoring': {
            'score_type': 'ranking_score_not_predicted_stars',
            'top_k': model.TOP_K,
            'popularity_weight': model.NEW_USER_POPULARITY_WEIGHT,
            'collaborative_weight_if_trained': model.ESTABLISHED_COLLABORATIVE_WEIGHT,
            'genre_weight_if_selected': model.GENRE_WEIGHT if genres else 0.0,
        },
        'items': [{**movie_details(model, model.id_to_col[r['tmdb_id']]),
                   'score': r['score'], 'collaborative_weight': r['collaborative_weight'],
                   'genre_match': r['genre_match'], 'strategy': r['strategy']}
                  for r in results]
    }
