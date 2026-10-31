import zipfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest
from httpx import ASGITransport, AsyncClient

from eva_agent.config.settings import WebSettings
from eva_agent.server.sandbox import SandboxClient, SandboxRejected, SandboxUnavailable
from eva_agent.server.state import AppState
from eva_agent.storage.uploads import UPLOAD_EXTRACTED, UPLOAD_FAILED
from tests.support.api import csrf_headers, login
from tests.support.runtime import fake_config
from tests.support.server import build_test_app


def _package(tmp_path: Path, members: dict[str, bytes]) -> bytes:
    path = tmp_path / "package.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path.read_bytes()


async def _upload(client: AsyncClient, data: bytes, filename: str = "package.zip"):
    csrf = await login(client)
    return await client.post(
        "/api/uploads",
        files={"file": (filename, data, "application/zip")},
        headers=csrf_headers(csrf),
    )


async def test_upload_extracts_and_reports_manifest(client: AsyncClient, tmp_path) -> None:
    response = await _upload(client, _package(tmp_path, {"a.txt": b"hello", "dir/b.txt": b"world"}))

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "extracted"
    upload_id = body["upload_id"]
    detail = await client.get(f"/api/uploads/{upload_id}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["status"] == UPLOAD_EXTRACTED
    assert [item["path"] for item in payload["manifest"]["files"]] == ["a.txt", "dir/b.txt"]


async def test_upload_requires_csrf(client: AsyncClient, tmp_path) -> None:
    await login(client)
    response = await client.post(
        "/api/uploads",
        files={"file": ("package.zip", _package(tmp_path, {"a.txt": b"x"}), "application/zip")},
    )
    assert response.status_code == 403


async def test_upload_rejects_traversal_and_keeps_no_files(client: AsyncClient, state: AppState, tmp_path) -> None:
    data = _package(tmp_path, {"../evil.txt": b"x"})

    response = await _upload(client, data)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upload_rejected"
    uploads_root = Path(state.config.storage.data_dir) / "uploads"
    roots = list(uploads_root.iterdir())
    assert len(roots) == 1
    assert not (roots[0] / "extracted").exists()


async def test_upload_rejects_oversized_body(tmp_path) -> None:
    from httpx import ASGITransport
    from httpx import AsyncClient as LimitedClient

    from eva_agent.config.settings import WebSettings
    from tests.support.runtime import fake_config
    from tests.support.server import build_test_app

    config = fake_config(tmp_path).model_copy(update={"web": WebSettings(upload_max_bytes=10)})
    app, limited_state = await build_test_app(tmp_path, config=config)
    try:
        async with LimitedClient(
            transport=ASGITransport(app=app), base_url="https://test"
        ) as limited:
            response = await _upload(limited, b"x" * 100)
        assert response.status_code == 413
        uploads_root = Path(limited_state.config.storage.data_dir) / "uploads"
        assert not uploads_root.exists() or not any(uploads_root.iterdir())
    finally:
        await limited_state.engine.dispose()


async def test_upload_rejects_unknown_format(client: AsyncClient) -> None:
    response = await _upload(client, b"plain text")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upload_rejected"


async def test_retry_unknown_upload_is_404(client: AsyncClient) -> None:
    csrf = await login(client)
    response = await client.post("/api/uploads/upload_missing/retry", headers=csrf_headers(csrf))
    assert response.status_code == 404


async def test_rejected_upload_is_recorded_as_failed(client: AsyncClient, state: AppState, tmp_path) -> None:
    response = await _upload(client, _package(tmp_path, {"../evil.txt": b"x"}))
    assert response.status_code == 400
    uploads_root = Path(state.config.storage.data_dir) / "uploads"
    assert len(list(uploads_root.iterdir())) == 1
    upload_id = next(uploads_root.iterdir()).name
    detail = await client.get(f"/api/uploads/{upload_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == UPLOAD_FAILED


async def test_retry_reenqueues_only_failed_uploads(client: AsyncClient, state: AppState, tmp_path) -> None:
    rejected = await _upload(client, _package(tmp_path, {"../evil.txt": b"x"}))
    assert rejected.status_code == 400
    upload_id = next((Path(state.config.storage.data_dir) / "uploads").iterdir()).name
    accepted = await _upload(client, _package(tmp_path, {"ok.txt": b"x"}))
    accepted_id = accepted.json()["upload_id"]

    # A fresh login is the session that will send the retry requests.
    csrf = await login(client)
    retry = await client.post(f"/api/uploads/{upload_id}/retry", headers=csrf_headers(csrf))
    assert retry.status_code == 400
    retry = await client.post(f"/api/uploads/{accepted_id}/retry", headers=csrf_headers(csrf))
    assert retry.status_code == 409


async def test_get_and_retry_require_a_session(client: AsyncClient) -> None:
    detail = await client.get("/api/uploads/upload_missing")
    retry = await client.post("/api/uploads/upload_missing/retry")

    assert detail.status_code == 401
    assert retry.status_code == 401


async def test_upload_sanitizes_the_client_filename(client: AsyncClient, tmp_path) -> None:
    response = await _upload(
        client,
        _package(tmp_path, {"a.txt": b"x"}),
        filename="../../evil.zip",
    )

    assert response.status_code == 202
    detail = await client.get(f"/api/uploads/{response.json()['upload_id']}")
    assert detail.json()["filename"] == "evil.zip"


class _UnavailableSandbox:
    async def extract(self, upload_id: str, format: str) -> dict:
        raise SandboxUnavailable("connection to 10.0.0.1 refused")


class _RejectingSandbox:
    async def extract(self, upload_id: str, format: str) -> dict:
        raise SandboxRejected("path_traversal: /etc/passwd is secret")


class _FlakySandbox:
    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, upload_id: str, format: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("raw upstream text /secret")
        return {"format": format, "files": [], "total_bytes": 0}


async def _with_sandbox(tmp_path: Path, sandbox: object):
    return await build_test_app(tmp_path, sandbox=cast(SandboxClient, sandbox))


async def test_sandbox_unavailable_marks_upload_failed(tmp_path: Path) -> None:
    app, state = await _with_sandbox(tmp_path, _UnavailableSandbox())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
            response = await _upload(client, _package(tmp_path, {"a.txt": b"x"}))
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "unavailable"
            upload_id = next((Path(state.config.storage.data_dir) / "uploads").iterdir()).name
            detail = await client.get(f"/api/uploads/{upload_id}")
        assert detail.json()["status"] == UPLOAD_FAILED
        assert detail.json()["error"] == "sandbox_unavailable"
        assert "10.0.0.1" not in detail.text
    finally:
        await state.engine.dispose()


async def test_rejection_stores_only_the_allowlisted_reason(tmp_path: Path) -> None:
    app, state = await _with_sandbox(tmp_path, _RejectingSandbox())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
            response = await _upload(client, _package(tmp_path, {"a.txt": b"x"}))
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "upload_rejected"
            upload_id = next((Path(state.config.storage.data_dir) / "uploads").iterdir()).name
            detail = await client.get(f"/api/uploads/{upload_id}")
        assert detail.json()["status"] == UPLOAD_FAILED
        assert detail.json()["error"] == "rejected"
        assert "/etc/passwd" not in detail.text
    finally:
        await state.engine.dispose()


async def test_unexpected_sandbox_error_marks_failed_and_allows_retry(tmp_path: Path) -> None:
    sandbox = _FlakySandbox()
    app, state = await _with_sandbox(tmp_path, sandbox)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
            response = await _upload(client, _package(tmp_path, {"a.txt": b"x"}))
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "unavailable"
            assert "/secret" not in response.text
            upload_id = next((Path(state.config.storage.data_dir) / "uploads").iterdir()).name
            detail = await client.get(f"/api/uploads/{upload_id}")
            assert detail.json()["status"] == UPLOAD_FAILED
            assert detail.json()["error"] == "sandbox_unavailable"

            csrf = await login(client)
            retry = await client.post(f"/api/uploads/{upload_id}/retry", headers=csrf_headers(csrf))
        assert retry.status_code == 202
        assert sandbox.calls == 2
    finally:
        await state.engine.dispose()


async def test_failed_create_removes_the_upload_directory(
    client: AsyncClient, state: AppState, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(state.uploads, "create", boom)

    with pytest.raises(RuntimeError):
        await _upload(client, _package(tmp_path, {"a.txt": b"x"}))

    uploads_root = Path(state.config.storage.data_dir) / "uploads"
    assert not uploads_root.exists() or not any(uploads_root.iterdir())


_CHUNK = 64 * 1024
_BOUNDARY = "evaboundary"


def _chunked_multipart(total: int) -> AsyncIterator[bytes]:
    async def body() -> AsyncIterator[bytes]:
        yield f"--{_BOUNDARY}\r\n".encode()
        yield (
            b'Content-Disposition: form-data; name="file"; filename="package.zip"\r\n'
            b"Content-Type: application/zip\r\n\r\n"
        )
        sent = 0
        while sent < total:
            size = min(_CHUNK, total - sent)
            yield b"x" * size
            sent += size
        yield f"\r\n--{_BOUNDARY}--\r\n".encode()

    return body()


async def test_upload_rejects_oversized_chunked_body(tmp_path: Path) -> None:
    config = fake_config(tmp_path).model_copy(update={"web": WebSettings(upload_max_bytes=1024)})
    app, state = await build_test_app(tmp_path, config=config)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
            csrf = await login(client)
            response = await client.post(
                "/api/uploads",
                content=_chunked_multipart(1_114_112),
                headers={
                    **csrf_headers(csrf),
                    "content-type": f"multipart/form-data; boundary={_BOUNDARY}",
                },
            )
        assert response.status_code == 413, response.text
        assert response.json()["error"]["code"] == "upload_rejected"
        uploads_root = Path(state.config.storage.data_dir) / "uploads"
        assert not uploads_root.exists() or not any(uploads_root.iterdir())
    finally:
        await state.engine.dispose()


async def test_upload_size_limit_allows_normal_uploads(tmp_path: Path) -> None:
    config = fake_config(tmp_path).model_copy(update={"web": WebSettings(upload_max_bytes=1024)})
    app, state = await build_test_app(tmp_path, config=config)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
            response = await _upload(client, _package(tmp_path, {"a.txt": b"x"}))
        assert response.status_code == 202, response.text
    finally:
        await state.engine.dispose()
