from httpx import AsyncClient


async def test_health_reports_ok(client: AsyncClient) -> None:
    response = await client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "4"}


async def test_unknown_api_path_uses_the_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_bare_api_path_uses_the_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/api")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_spa_route_serves_index_html(client: AsyncClient) -> None:
    response = await client.get("/runs/run_example")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
