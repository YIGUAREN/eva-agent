import httpx
import pytest

from eva_agent.adapters.aig.client import AigClient
from eva_agent.adapters.aig.errors import AigUnavailable
from eva_agent.config.settings import AigSettings


def _client(handler, **settings) -> tuple[AigClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://aig.local")
    return AigClient(AigSettings(**settings), http=http), http


@pytest.mark.asyncio
async def test_upload_posts_the_file_field_and_returns_file_url() -> None:
    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/app/taskapi/upload"
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "status": 0,
                "message": "文件上传成功",
                "data": {"filename": "pkg.zip", "fileUrl": "pkg_1793423249288665728_1793423249.zip"},
            },
        )

    client, http = _client(handler)
    async with http:
        assert await client.upload("pkg.zip", b"PK\x03\x04") == "pkg_1793423249288665728_1793423249.zip"
    assert b'name="file"' in captured["body"]
    assert b"PK\x03\x04" in captured["body"]


@pytest.mark.asyncio
async def test_upload_rejects_a_file_url_that_looks_like_a_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "message": "ok", "data": {"fileUrl": "../../etc/passwd"}})

    client, http = _client(handler)
    async with http:
        with pytest.raises(AigUnavailable, match="invalid fileUrl"):
            await client.upload("pkg.zip", b"PK\x03\x04")


@pytest.mark.parametrize("file_url", [".", ".."])
@pytest.mark.asyncio
async def test_upload_rejects_a_dot_only_file_url(file_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "message": "ok", "data": {"fileUrl": file_url}})

    client, http = _client(handler)
    async with http:
        with pytest.raises(AigUnavailable, match="invalid fileUrl"):
            await client.upload("pkg.zip", b"PK\x03\x04")


@pytest.mark.asyncio
async def test_upload_maps_transport_failures_without_echoing_the_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret-token-in-message")

    client, http = _client(handler)
    async with http:
        with pytest.raises(AigUnavailable) as excinfo:
            await client.upload("pkg.zip", b"PK\x03\x04")
    assert "secret-token-in-message" not in str(excinfo.value)
