from test_backend import client  # Reuse synthetic model + temporary database fixture


def test_password_registration_login_and_session_rotation(client):
    payload = {'username': 'frontend_user', 'password': 'long-password123'}
    assert client.post('/auth/register', json={'username': 'invalid', 'password': 'short'}).status_code == 422
    created = client.post('/auth/register', json=payload)
    assert created.status_code == 201
    token = created.json()['token']
    assert client.post('/auth/register', json=payload).status_code == 409
    assert client.post('/auth/login', json={'username': 'frontend_user', 'password': 'incorrect'}).status_code == 401
    logged = client.post('/auth/login', json=payload)
    assert logged.status_code == 200
    new_token = logged.json()['token']
    assert new_token != token
    assert client.get('/users/me', headers={'Authorization': f'Bearer {token}'}).status_code == 401
    auth = {'Authorization': f'Bearer {new_token}'}
    assert client.get('/users/me', headers=auth).json()['username'] == 'frontend_user'
    assert client.post('/auth/logout', headers=auth).status_code == 204
    assert client.get('/users/me', headers=auth).status_code == 401


def test_watchlist_persists_and_rating_removes_it(client):
    user = client.post('/auth/register', json={'username': 'watchlist_user', 'password': 'password123'}).json()
    auth = {'Authorization': f"Bearer {user['token']}"}
    assert client.post('/watchlist', json={'tmdb_id': 11}).status_code == 401
    assert client.post('/watchlist', headers=auth, json={'tmdb_id': 999}).status_code == 404
    assert client.post('/watchlist', headers=auth, json={'tmdb_id': 11}).status_code == 201
    assert client.post('/watchlist', headers=auth, json={'tmdb_id': 11}).status_code == 201
    entries = client.get('/watchlist', headers=auth).json()['items']
    assert len(entries) == 1 and entries[0]['title'] == 'Galaxy Film'
    assert client.put('/ratings', headers=auth, json={'tmdb_id': 11, 'rating': 4.5}).status_code == 200
    assert client.get('/watchlist', headers=auth).json()['items'] == []
    assert client.get('/ratings', headers=auth).json()[0]['rating'] == 4.5
    assert client.delete('/watchlist/11', headers=auth).status_code == 204


def test_legacy_users_cannot_password_login(client):
    assert client.post('/users', json={'username': 'legacy_user'}).status_code == 201
    assert client.post('/auth/login', json={'username': 'legacy_user', 'password': 'password123'}).status_code == 401
