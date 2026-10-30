"""Reports, downloadable artifacts, and public catalogs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from eva_agent.adapters.openjudge.catalog import CatalogError, load_catalog, summarize
from eva_agent.config.settings import ConfigError
from eva_agent.config.targets import load_targets
from eva_agent.runtime.artifacts import run_paths
from eva_agent.server.app import get_state
from eva_agent.server.auth import current_session
from eva_agent.server.errors import ApiError
from eva_agent.server.routes.runs import _load_public_run, _load_run
from eva_agent.server.schemas import redact_absolute_paths, sanitize_public_payload
from eva_agent.storage.models import AdminSessionRow

router = APIRouter(prefix="/api", tags=["reports"])

_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]{1,255}")

_BUDGET_LINE_PREFIX = "- 预算："


def _redact_data_dir(text: str, data_dir: str) -> str:
    resolved = str(Path(data_dir).resolve())
    return text.replace(resolved, "<data>").replace(str(data_dir), "<data>")


def _sanitize_public_markdown(markdown: str, data_dir: str) -> str:
    """Sanitize report markdown for the public route only.

    Admin responses return the stored markdown unchanged. Public reports must
    never reveal absolute filesystem paths or budget figures.
    """
    text = redact_absolute_paths(_redact_data_dir(markdown, data_dir))
    return "\n".join(line for line in text.split("\n") if not line.startswith(_BUDGET_LINE_PREFIX))


async def _report(state, run_id: str, *, public: bool = False) -> dict[str, object]:
    snapshot = await state.reports.get(run_id)
    if snapshot is None:
        raise ApiError(404, "report_not_ready", "report is not ready")
    payload = snapshot.payload
    digest = snapshot.sha256
    markdown = snapshot.markdown
    if public:
        payload = sanitize_public_payload(snapshot.payload)
        # The stored digest covers the unsanitized payload, so recompute it with
        # the same canonicalization used by ReportRepository.upsert.
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        markdown = _sanitize_public_markdown(markdown, state.config.storage.data_dir)
    return {
        "run_id": snapshot.run_id,
        "version": snapshot.version,
        "sha256": digest,
        "markdown": markdown,
        "report": payload,
    }


async def _artifact(state, run_id: str, name: str, *, public: bool) -> FileResponse:
    if _SAFE_NAME.fullmatch(name) is None:
        raise ApiError(404, "not_found", "artifact not found")
    row = await state.artifacts.get(run_id, name)
    if row is None or (public and row.visibility != "public"):
        raise ApiError(404, "not_found", "artifact not found")
    path = run_paths(state.config.storage.data_dir, run_id).root / row.name
    if not path.is_file():
        raise ApiError(404, "not_found", "artifact not found")
    return FileResponse(path, media_type=row.content_type, filename=row.name)


@router.get("/runs/{run_id}/report")
async def admin_report(
    run_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
) -> dict[str, object]:
    state = get_state(request)
    await _load_run(state, run_id)
    return await _report(state, run_id)


@router.get("/runs/{run_id}/artifacts/{name}")
async def admin_artifact(
    run_id: str,
    name: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
) -> FileResponse:
    state = get_state(request)
    await _load_run(state, run_id)
    return await _artifact(state, run_id, name, public=False)


@router.get("/public/runs/{run_id}/report")
async def public_report(run_id: str, request: Request) -> dict[str, object]:
    state = get_state(request)
    await _load_public_run(state, run_id)
    return await _report(state, run_id, public=True)


@router.get("/public/runs/{run_id}/artifacts/{name}")
async def public_artifact(run_id: str, name: str, request: Request) -> FileResponse:
    state = get_state(request)
    await _load_public_run(state, run_id)
    return await _artifact(state, run_id, name, public=True)


@router.get("/graders")
async def graders(request: Request) -> dict[str, list[dict[str, object]]]:
    state = get_state(request)
    try:
        catalog = load_catalog(Path(state.config.openjudge.catalog_path))
    except CatalogError:
        raise ApiError(503, "unavailable", "grader catalog unavailable") from None
    return {"graders": summarize(catalog)}


@router.get("/demo-targets")
async def demo_targets(request: Request) -> dict[str, list[dict[str, str]]]:
    state = get_state(request)
    try:
        targets = load_targets(Path(state.config.target.targets_path))
    except ConfigError:
        raise ApiError(503, "unavailable", "target catalog unavailable") from None
    return {
        "targets": [
            {"name": item.name, "kind": item.kind.value, "provider": item.provider, "model": item.model}
            for item in targets.values()
        ]
    }
