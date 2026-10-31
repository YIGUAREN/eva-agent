import zipfile
from pathlib import Path

from httpx import AsyncClient

from eva_agent.domain.run import RunStatus
from eva_agent.packages.manifest import FileEntry, PackageManifest
from eva_agent.server.routes import packages as packages_route
from eva_agent.server.state import AppState
from tests.support.api import PASTE_PAYLOAD, csrf_headers, login


def _manifest(run_id: str, *, replayable: bool = True) -> PackageManifest:
    return PackageManifest(
        kind="grader_package",
        run_id=run_id,
        created_at="2026-10-31T00:00:00+00:00",
        report_version=4,
        replayable=replayable,
        replay_blockers=[] if replayable else ["quality was truncated"],
        files={"inputs/rows.json": FileEntry(sha256="a" * 64, size=3)},
    )


async def _completed_run(client: AsyncClient, state: AppState) -> str:
    csrf = await login(client)
    response = await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))
    run_id = response.json()["run_id"]
    await state.runs.set_status(run_id, RunStatus.COMPLETED)
    return run_id


def _patch_builder(monkeypatch, *, replayable: bool = True) -> list[str]:
    calls: list[str] = []

    async def build(*, config, run_row, out_dir: Path):
        calls.append(run_row.id)
        archive = Path(out_dir) / f"{run_row.id}.grader-package.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("manifest.json", b"{}")
        return archive, _manifest(run_row.id, replayable=replayable)

    monkeypatch.setattr(packages_route, "build_grader_package", build)
    return calls


async def test_export_builds_stores_and_downloads_a_package(
    client: AsyncClient, state: AppState, monkeypatch
) -> None:
    run_id = await _completed_run(client, state)
    calls = _patch_builder(monkeypatch)
    csrf = await login(client)

    created = await client.post(f"/api/runs/{run_id}/grader-package", headers=csrf_headers(csrf))

    assert created.status_code == 201
    body = created.json()
    assert calls == [run_id]
    assert body["run_id"] == run_id
    assert body["replayable"] is True
    assert body["members"] == ["inputs/rows.json"]
    stored = await state.packages.latest(run_id)
    assert stored is not None
    assert stored.sha256 == body["sha256"]

    listed = await client.get(f"/api/runs/{run_id}/grader-package")
    assert listed.status_code == 200
    assert listed.json()["sha256"] == body["sha256"]

    download = await client.get(f"/api/runs/{run_id}/grader-package/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert download.content[:2] == b"PK"


async def test_download_rejects_a_tampered_archive(
    client: AsyncClient, state: AppState, monkeypatch
) -> None:
    run_id = await _completed_run(client, state)
    _patch_builder(monkeypatch)
    csrf = await login(client)
    created = await client.post(f"/api/runs/{run_id}/grader-package", headers=csrf_headers(csrf))
    assert created.status_code == 201
    archive = packages_route.package_archive_path(state.config.storage.data_dir, run_id)
    archive.write_bytes(b"PK tampered payload")

    response = await client.get(f"/api/runs/{run_id}/grader-package/download")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "package_corrupt"


async def test_export_reports_blockers_for_an_unreplayable_run(
    client: AsyncClient, state: AppState, monkeypatch
) -> None:
    run_id = await _completed_run(client, state)
    _patch_builder(monkeypatch, replayable=False)
    csrf = await login(client)
    body = (
        await client.post(f"/api/runs/{run_id}/grader-package", headers=csrf_headers(csrf))
    ).json()
    assert body["replayable"] is False
    assert body["replay_blockers"] == ["quality was truncated"]


async def test_export_requires_csrf(client: AsyncClient, state: AppState, monkeypatch) -> None:
    run_id = await _completed_run(client, state)
    _patch_builder(monkeypatch)
    await login(client)
    response = await client.post(f"/api/runs/{run_id}/grader-package")
    assert response.status_code == 403


async def test_export_rejects_a_run_that_is_not_completed(
    client: AsyncClient, state: AppState, monkeypatch
) -> None:
    csrf = await login(client)
    run_id = (await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))).json()["run_id"]
    _patch_builder(monkeypatch)
    response = await client.post(f"/api/runs/{run_id}/grader-package", headers=csrf_headers(csrf))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_completed"


async def test_unknown_run_is_not_found(client: AsyncClient, monkeypatch) -> None:
    _patch_builder(monkeypatch)
    csrf = await login(client)
    response = await client.post("/api/runs/run-missing/grader-package", headers=csrf_headers(csrf))
    assert response.status_code == 404


async def test_download_without_a_package_is_not_found(
    client: AsyncClient, state: AppState, monkeypatch
) -> None:
    run_id = await _completed_run(client, state)
    await login(client)
    response = await client.get(f"/api/runs/{run_id}/grader-package/download")
    assert response.status_code == 404
