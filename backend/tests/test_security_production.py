"""HTTP security behavior when running behind an HTTPS reverse proxy."""
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from test_backend import client

ORIGIN = 'https://reelwise.example.com'


def prod_client(monkeypatch):
    import backend
    monkeypatch.setenv('APP_ENV', 'production')
    monkeypatch.setenv('FRONTEND_ORIGIN', ORIGIN)
    return TestClient(backend.app, base_url=ORIGIN)


def test_secure_cookie_csrf_and_logout(client, monkeypatch):
    with prod_client(monkeypatch) as browser:
        origin = {'Origin': ORIGIN}
        forbidden = browser.post('/auth/register', json={'username': 'produser', 'password': 'longpassword123'})
        assert forbidden.status_code == 403
        result = browser.post('/auth/register', json={'username': 'produser', 'password': 'longpassword123'}, headers=origin)
        assert result.status_code == 201
        assert 'token' not in result.json()
        assert 'httponly' in result.headers['set-cookie'].lower()
        assert 'secure' in result.headers['set-cookie'].lower()
        assert browser.get('/auth/session').json()['username'] == 'produser'
        assert browser.put('/users/me/genres', json={'genres': ['Comedy']}, headers=origin).status_code == 403
        csrf = browser.cookies['reelwise_csrf']
        headers = {'Origin': ORIGIN, 'X-CSRF-Token': csrf}
        assert browser.put('/users/me/genres', json={'genres': ['Comedy']}, headers=headers).status_code == 200
        assert browser.post('/auth/logout', headers=headers).status_code == 204
        assert browser.get('/auth/session').status_code == 401


def test_authentication_expiration_and_legacy_registration(client, monkeypatch):
    import backend
    with prod_client(monkeypatch) as browser:
        assert browser.post('/users', json={'username': 'legacyblocked'}, headers={'Origin': ORIGIN}).status_code == 404
        result = browser.post('/auth/register', json={'username': 'expirationtest', 'password': 'longpassword123'}, headers={'Origin': ORIGIN})
        assert result.status_code == 201
        with backend.next_connection() as db:
            db.execute('UPDATE users SET token_expires_at=? WHERE username=?',
                ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), 'expirationtest'))
            db.commit()
        assert browser.get('/users/me').status_code == 401


def test_rate_limiting_works_in_production(client, monkeypatch):
    import backend
    backend._auth_attempts.clear()
    monkeypatch.setenv('AUTH_RATE_LIMIT_ATTEMPTS', '2')
    with prod_client(monkeypatch) as browser:
        headers = {'Origin': ORIGIN}
        body = {'username': 'notregistered', 'password': 'wrongpassword123'}
        assert browser.post('/auth/login', json=body, headers=headers).status_code == 401
        assert browser.post('/auth/login', json=body, headers=headers).status_code == 401
        assert browser.post('/auth/login', json=body, headers=headers).status_code == 429
    backend._auth_attempts.clear()


def test_frontend_uses_cookie_auth_not_browser_auth_storage():
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / 'frontend' / 'src'
    client_source = (src / 'api' / 'client.ts').read_text()
    app_source = (src / 'App.tsx').read_text()
    assert "credentials: 'same-origin'" in client_source
    assert 'X-CSRF-Token' in client_source
    assert 'Authorization' not in client_source
    assert 'localStorage' not in client_source
    assert 'sessionStorage' not in client_source
    assert 'setToken(' not in app_source
