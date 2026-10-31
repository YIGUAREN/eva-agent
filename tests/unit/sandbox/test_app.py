import zipfile

import httpx
import pytest

from eva_agent.config.settings import UploadsSettings
from eva_agent.runtime.uploads import upload_paths
from eva_agent.sandbox.app import create_sandbox_app

TOKEN = "sandbox-test-token"


def _app(tmp_path):
    return create_sandbox_app(data_dir=str(tmp_path), token=TOKEN, uploads=UploadsSettings())


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sandbox")


async def test_health_is_public(tmp_path) -> None:
    async with _client(_app(tmp_path)) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_extract_requires_the_token(tmp_path) -> None:
    async with _client(_app(tmp_path)) as client:
        response = await client.post("/extract", json={"upload_id": "upload_a", "format": "zip"})
    assert response.status_code == 401


async def test_extract_writes_under_the_upload_root(tmp_path) -> None:
    paths = upload_paths(tmp_path, "upload_a")
    paths.root.mkdir(parents=True)
    with zipfile.ZipFile(paths.package, "w") as archive:
        archive.writestr("a.txt", b"hello")

    async with _client(_app(tmp_path)) as client:
        response = await client.post(
            "/extract",
            json={"upload_id": "upload_a", "format": "zip", "relative_path": "package.bin"},
            headers={"X-Sandbox-Token": TOKEN},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "zip"
    assert body["manifest"]["files"][0]["path"] == "a.txt"
    assert (paths.extracted / "a.txt").read_bytes() == b"hello"


@pytest.mark.parametrize("relative_path", ["../package.bin", "/etc/passwd", "other.bin", "a/../../b"])
async def test_extract_rejects_foreign_relative_paths(tmp_path, relative_path: str) -> None:
    async with _client(_app(tmp_path)) as client:
        response = await client.post(
            "/extract",
            json={"upload_id": "upload_a", "format": "zip", "relative_path": relative_path},
            headers={"X-Sandbox-Token": TOKEN},
        )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "upload_rejected"


async def test_extract_rejects_unsafe_archive_without_leftovers(tmp_path) -> None:
    paths = upload_paths(tmp_path, "upload_a")
    paths.root.mkdir(parents=True)
    with zipfile.ZipFile(paths.package, "w") as archive:
        archive.writestr("../evil.txt", b"x")

    async with _client(_app(tmp_path)) as client:
        response = await client.post(
            "/extract",
            json={"upload_id": "upload_a", "format": "zip"},
            headers={"X-Sandbox-Token": TOKEN},
        )

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "path_traversal"
    assert not paths.extracted.exists()


async def test_extract_rejects_missing_package(tmp_path) -> None:
    async with _client(_app(tmp_path)) as client:
        response = await client.post(
            "/extract",
            json={"upload_id": "upload_missing", "format": "zip"},
            headers={"X-Sandbox-Token": TOKEN},
        )
    assert response.status_code == 400
