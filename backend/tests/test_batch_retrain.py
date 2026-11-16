import sqlite3
from pathlib import Path

import pytest

from batch_retrain import rating_fingerprint, retrain_once


def setup_files(tmp_path):
    db = tmp_path / 'app.sqlite3'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE ratings(user_id INTEGER, tmdb_id INTEGER, rating REAL)')
        conn.execute('INSERT INTO ratings VALUES (1, 12, 4.5)')
    data = tmp_path / 'processed'
    data.mkdir()
    (data / 'training_movies.csv').write_text('id,title\n12,Example\n')
    (data / 'training_ratings.csv').write_text('userId,movieId,rating\n1,1,4.5\n')
    return db, data, tmp_path / 'hybrid.joblib', tmp_path / 'state.json'


def test_retraining_skips_unchanged_and_updates_after_rating(tmp_path):
    db, data, model, state = setup_files(tmp_path)
    calls = []

    def fake_runner(command, cwd, check):
        calls.append(command)
        assert check is True
        assert '--skip-evaluation' in command
        Path(command[command.index('--output') + 1]).write_bytes(b'candidate-model-' + str(len(calls)).encode())

    opts = dict(db=db, data=data, model=model, state=state, runner=fake_runner)
    assert retrain_once(**opts) == 'updated'
    assert model.read_bytes() == b'candidate-model-1'
    assert retrain_once(**opts) == 'unchanged'
    assert len(calls) == 1
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE ratings SET rating=5.0')
    assert retrain_once(**opts) == 'updated'
    assert model.read_bytes() == b'candidate-model-2'
    assert len(calls) == 2
    assert not (tmp_path / 'hybrid.joblib.retrain.lock').exists()


def test_failed_training_keeps_previous_model_and_state(tmp_path):
    db, data, model, state = setup_files(tmp_path)
    model.write_bytes(b'previous-good-model')

    def failing_runner(command, cwd, check):
        Path(command[command.index('--output') + 1]).write_bytes(b'broken')
        raise RuntimeError('training failed')

    with pytest.raises(RuntimeError, match='training failed'):
        retrain_once(db=db, data=data, model=model, state=state, runner=failing_runner)
    assert model.read_bytes() == b'previous-good-model'
    assert not state.exists()
    assert list(tmp_path.glob('.hybrid-training-*')) == []
    assert not (tmp_path / 'hybrid.joblib.retrain.lock').exists()


def test_empty_database_does_not_trigger_unnecessary_training(tmp_path):
    db, data, model, state = setup_files(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute('DELETE FROM ratings')
    assert rating_fingerprint(db)[1] == 0
    assert retrain_once(db=db, data=data, model=model, state=state) == 'no-ratings'


def test_lock_prevents_parallel_training(tmp_path):
    db, data, model, state = setup_files(tmp_path)
    (tmp_path / 'hybrid.joblib.retrain.lock').write_text('existing job')
    assert retrain_once(db=db, data=data, model=model, state=state) == 'locked'
