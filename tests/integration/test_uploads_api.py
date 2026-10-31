import zipfile
from pathlib import Path

from httpx import AsyncClient

from eva_agent.server.state import AppState
from eva_agent.storage.uploads import UPLOAD_EXTRACTED, UPLOAD_FAILED
from tests.support.api import csrf_headers, login


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
