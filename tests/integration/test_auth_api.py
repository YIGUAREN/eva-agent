from httpx import AsyncClient

from eva_agent.server.auth import SESSION_COOKIE
from tests.support.server import TEST_PASSWORD


async def _login(client: AsyncClient, password: str = TEST_PASSWORD) -> str:
    response = await client.post("/api/auth/login", json={"password": password})
    assert response.status_code == 200, response.text
    return response.json()["csrf_token"]


async def test_login_sets_an_http_only_cookie(client: AsyncClient) -> None:
    csrf = await _login(client)

    header = client.cookies
    assert header.get(SESSION_COOKIE)
    assert len(csrf) >= 32


async def test_login_cookie_flags_are_secure(client: AsyncClient) -> None:
    response = await client.post("/api/auth/login", json={"password": TEST_PASSWORD})

    set_cookie = response.headers["set-cookie"]
    assert "HttpOnly" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Secure" in set_cookie
    assert "Path=/" in set_cookie


async def test_wrong_password_is_rejected(client: AsyncClient) -> None:
    response = await client.post("/api/auth/login", json={"password": "wrong password"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"
    assert SESSION_COOKIE not in response.cookies


async def test_login_is_rate_limited(client: AsyncClient) -> None:
    for _ in range(5):
        await client.post("/api/auth/login", json={"password": "wrong password"})

    blocked = await client.post("/api/auth/login", json={"password": TEST_PASSWORD})

    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "rate_limited"


async def test_me_requires_a_session(client: AsyncClient) -> None:
    response = await client.get("/api/auth/me")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


async def test_me_returns_the_csrf_token(client: AsyncClient) -> None:
    csrf = await _login(client)

    response = await client.get("/api/auth/me")

    assert response.status_code == 200
    assert response.json()["csrf_token"] == csrf


async def test_logout_requires_csrf_and_revokes_the_session(client: AsyncClient) -> None:
    csrf = await _login(client)

    rejected = await client.post("/api/auth/logout")
    assert rejected.status_code == 403
    assert rejected.json()["error"]["code"] == "csrf_failed"

    accepted = await client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
    assert accepted.status_code == 204
    assert "Path=/" in accepted.headers["set-cookie"]
    assert (await client.get("/api/auth/me")).status_code == 401


async def test_login_body_is_validated(client: AsyncClient) -> None:
    response = await client.post("/api/auth/login", json={})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["fields"] == ["body.password"]
