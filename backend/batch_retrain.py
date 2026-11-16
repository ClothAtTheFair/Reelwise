"""Safely retrain the Reelwise model on saved app ratings.

Use --once from Windows Task Scheduler/cron, or --daemon to run a long-lived
scheduler. Nothing retrains inside a FastAPI request. Retrains only when the
app ratings have changed since the last *successful* scheduled run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def rating_fingerprint(db_path: Path) -> tuple[str, int]:
    """Hash the current app ratings in stable order (no personally identifying output)."""
    if not db_path.is_file():
        return '', 0
    connection = sqlite3.connect(f'file:{db_path.resolve().as_posix()}?mode=ro', uri=True, timeout=15)
    digest = hashlib.sha256()
    try:
        rows = connection.execute(
            'SELECT user_id, tmdb_id, rating FROM ratings ORDER BY user_id, tmdb_id'
        )
        count = 0
        for user_id, movie_id, rating in rows:
            digest.update(f'{int(user_id)}:{int(movie_id)}:{float(rating):.6f}\n'.encode('ascii'))
            count += 1
        return digest.hexdigest(), count
    except sqlite3.OperationalError as exc:
        if 'no such table' in str(exc).lower():
            return '', 0
        raise
    finally:
        connection.close()


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.retrain-state-', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def retrain_once(*, db: Path, data: Path, model: Path, state: Path,
                 factors: int = 40, seed: int = 42, force: bool = False,
                 runner=subprocess.run) -> str:
    """Return one of: updated, unchanged, no-ratings, locked."""
    db, data, model, state = map(lambda p: Path(p).resolve(), (db, data, model, state))
    model.parent.mkdir(parents=True, exist_ok=True)
    lock_path = model.with_name(model.name + '.retrain.lock')
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return 'locked'
    try:
        with os.fdopen(lock_fd, 'w', encoding='utf-8') as lock_file:
            lock_file.write(f'pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n')
        fingerprint, count = rating_fingerprint(db)
        if count == 0 and not force:
            return 'no-ratings'
        previous = load_state(state)
        if not force and model.is_file() and fingerprint == previous.get('fingerprint'):
            return 'unchanged'
        if not (data / 'training_movies.csv').is_file() or not (data / 'training_ratings.csv').is_file():
            raise FileNotFoundError('Processed training CSVs missing; run filter_catalog.py first.')
        fd, candidate_name = tempfile.mkstemp(prefix='.hybrid-training-', suffix='.joblib', dir=model.parent)
        os.close(fd)
        candidate = Path(candidate_name)
        candidate.unlink()  # train.py creates it itself
        try:
            command = [sys.executable, str(ROOT / 'train.py'), '--data', str(data),
                       '--app-db', str(db), '--output', str(candidate),
                       '--skip-evaluation', '--factors', str(factors), '--seed', str(seed)]
            print(f'[retrain] Training on {count:,} saved application ratings...', flush=True)
            runner(command, cwd=ROOT, check=True)
            if not candidate.is_file() or candidate.stat().st_size == 0:
                raise RuntimeError('Training returned without producing a nonempty model file')
            # Shared group read permission allows the isolated API service to load it.
            candidate.chmod(0o640)
            # Atomic replacement keeps the previous model usable if training fails.
            os.replace(candidate, model)
            atomic_json(state, {
                'fingerprint': fingerprint,
                'rating_count_at_start': count,
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'model_path': str(model),
                'factors': factors,
                'seed': seed,
            })
            print(f'[retrain] Installed {model}. API reloads it on next model request.', flush=True)
            return 'updated'
        finally:
            candidate.unlink(missing_ok=True)
    finally:
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-db', type=Path, default=ROOT / 'data/app.sqlite3')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/hybrid.joblib')
    parser.add_argument('--state', type=Path, default=ROOT / 'models/retrain_state.json')
    parser.add_argument('--factors', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true', help='Train even if app ratings are unchanged or absent')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='Run once (default); suitable for daily OS scheduling')
    mode.add_argument('--daemon', action='store_true', help='Repeat on a timer until stopped')
    parser.add_argument('--interval-hours', type=float, default=24.0, help='Interval for --daemon (default 24)')
    args = parser.parse_args()
    if args.interval_hours <= 0:
        parser.error('--interval-hours must be positive')
    if args.factors <= 0:
        parser.error('--factors must be positive')
    while True:
        try:
            outcome = retrain_once(db=args.app_db, data=args.data, model=args.model,
                                   state=args.state, factors=args.factors,
                                   seed=args.seed, force=args.force)
            print(f'[retrain] {outcome}', flush=True)
        except Exception as exc:
            print(f'[retrain] ERROR: {exc}', file=sys.stderr, flush=True)
            if not args.daemon:
                raise SystemExit(1) from exc
        if not args.daemon:
            return
        try:
            time.sleep(args.interval_hours * 3600)
        except KeyboardInterrupt:
            print('[retrain] stopped')
            return


if __name__ == '__main__':
    main()
