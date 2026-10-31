import httpx
import pytest

from eva_agent.server.sandbox import SandboxClient, SandboxRejected, SandboxUnavailable


def _client(handler) -> SandboxClient:
    return SandboxClient(
        "http://sandbox",
        "secret-token",
        5.0,
        transport=httpx.MockTransport(handler),
    )


async def test_extract_returns_the_manifest() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Sandbox-Token"] == "secret-token"
        assert request.url.path == "/extract"
        return httpx.Response(
            200,
            json={
                "upload_id": "upload_a",
                "format": "zip",
                "manifest": {
                    "format": "zip",
                    "files": [{"path": "a.txt", "size": 1, "sha256": "a" * 64}],
                    "total_bytes": 1,
                },
            },
        )

    manifest = await _client(handler).extract("upload_a", "zip")
    assert manifest["files"][0]["path"] == "a.txt"


async def test_rejection_maps_to_a_safe_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": {"code": "upload_rejected", "message": "symlink"}})

    with pytest.raises(SandboxRejected) as caught:
        await _client(handler).extract("upload_a", "zip")
    assert caught.value.reason == "symlink"


async def test_unknown_reason_is_replaced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": {"code": "upload_rejected", "message": "C:/secret/x"}})

    with pytest.raises(SandboxRejected) as caught:
        await _client(handler).extract("upload_a", "zip")
    assert caught.value.reason == "rejected"


async def test_auth_failure_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": {"code": "forbidden", "message": "bad"}})

    with pytest.raises(SandboxUnavailable):
        await _client(handler).extract("upload_a", "zip")


async def test_transport_error_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(SandboxUnavailable):
        await _client(handler).extract("upload_a", "zip")


async def test_invalid_manifest_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"upload_id": "upload_a", "format": "zip", "manifest": {"files": "nope"}})

    with pytest.raises(SandboxUnavailable):
        await _client(handler).extract("upload_a", "zip")
