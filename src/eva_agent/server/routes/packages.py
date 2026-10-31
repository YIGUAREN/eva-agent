"""Administrator export of a completed run's grader package."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.run import RunStatus
from eva_agent.packages.grader_package import build_grader_package
from eva_agent.packages.manifest import PackageError, sha256_file
from eva_agent.runtime.artifacts import run_paths
from eva_agent.server.app import get_state
from eva_agent.server.auth import csrf_protected, current_session
from eva_agent.server.errors import ApiError
from eva_agent.server.schemas import redact_absolute_paths
from eva_agent.server.state import AppState
from eva_agent.storage.models import AdminSessionRow, GraderPackageRow

router = APIRouter(prefix="/api/runs", tags=["packages"])


def package_archive_path(data_dir: str | Path, run_id: str) -> Path:
    return run_paths(data_dir, run_id).root / "packages" / f"{run_id}.grader-package.zip"


def _view(row: GraderPackageRow, *, replayable: bool | None = None, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "package_id": row.id,
        "run_id": row.run_id,
        "sha256": row.sha256,
        "size_bytes": row.size_bytes,
        "created_at": row.created_at.isoformat(),
    }
    if replayable is not None:
        payload["replayable"] = replayable
    payload.update(extra)
    return payload


async def _run(state: AppState, run_id: str):
    row = await state.runs.get(run_id)
    if row is None:
        raise ApiError(404, "not_found", "run not found")
    return row


@router.post("/{run_id}/grader-package", status_code=201)
async def export_grader_package(
    run_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(csrf_protected)],
) -> dict[str, object]:
    state = get_state(request)
    run = await _run(state, run_id)
    if run.status != RunStatus.COMPLETED.value:
        raise ApiError(409, "not_completed", "only completed runs can be exported")
    try:
        archive, manifest = await build_grader_package(
            config=state.config,
            run_row=run,
            out_dir=package_archive_path(state.config.storage.data_dir, run_id).parent,
        )
    except PackageError as exc:
        # The builder's messages are user-visible; a failure can quote an OS
        # error whose text embeds an absolute path, which must never reach the
        # client.
        raise ApiError(409, "package_rejected", redact_absolute_paths(str(exc))) from None
    package_id = f"pkg_{uuid4().hex}"
    await state.packages.add(
        package_id=package_id,
        run_id=run_id,
        sha256=sha256_file(archive),
        size_bytes=archive.stat().st_size,
    )
    await state.audit.record(
        AuditEvent(
            run_id=run_id,
            actor="admin",
            action="run.export_package",
            outcome="success",
            authorization_ref=run.authorization_ref,
        )
    )
    row = await state.packages.latest(run_id)
    assert row is not None
    return _view(
        row,
        replayable=manifest.replayable,
        replay_blockers=list(manifest.replay_blockers),
        members=sorted(manifest.files),
    )


@router.get("/{run_id}/grader-package")
async def view_grader_package(
    run_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
) -> dict[str, object]:
    state = get_state(request)
    await _run(state, run_id)
    row = await state.packages.latest(run_id)
    archive = package_archive_path(state.config.storage.data_dir, run_id)
    if row is None or not archive.is_file():
        raise ApiError(404, "not_found", "no grader package for this run")
    return _view(row, download_url=f"/api/runs/{run_id}/grader-package/download")


@router.get("/{run_id}/grader-package/download")
async def download_grader_package(
    run_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
) -> FileResponse:
    state = get_state(request)
    await _run(state, run_id)
    row = await state.packages.latest(run_id)
    archive = package_archive_path(state.config.storage.data_dir, run_id)
    if row is None or not archive.is_file():
        raise ApiError(404, "not_found", "no grader package for this run")
    return FileResponse(
        archive, media_type="application/zip", filename=f"{run_id}.grader-package.zip"
    )
