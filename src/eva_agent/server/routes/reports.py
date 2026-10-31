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
from eva_agent.server.routes.runs import _internal_hosts, _load_public_run, _load_run
from eva_agent.server.schemas import (
    redact_internal_addresses_and_secrets,
    redact_markdown_paths,
    sanitize_public_payload,
)
from eva_agent.storage.models import AdminSessionRow

router = APIRouter(prefix="/api", tags=["reports"])

_SAFE_NAME = re.compile(r"[A-Za-z0-9._-]{1,255}")
_AIG_NAME = re.compile(r"aig/([A-Za-z0-9_-]{1,128})\.(json|csv)")

# Public Markdown never carries budget or spend figures.
_HIDDEN_LINE_PREFIXES = ("- 预算：", "- 已花费：")

_PUBLIC_USAGE_KEYS = ("estimated", "cached", "fresh", "budget_enforced")

_PUBLIC_SAMPLE_COUNT_KEYS = ("dimension", "sample_count", "completed_samples", "missing_samples")


def public_report_usage(payload: dict[str, object]) -> dict[str, object]:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return {}
    return {key: raw[key] for key in _PUBLIC_USAGE_KEYS if key in raw}


def public_truncated_sample_counts(payload: dict[str, object]) -> list[dict[str, object]]:
    """Whitelist each truncation entry so hostile extra keys never go public."""
    raw = payload.get("truncated_sample_counts")
    if not isinstance(raw, list):
        return []
    return [
        {key: item[key] for key in _PUBLIC_SAMPLE_COUNT_KEYS if key in item}
        for item in raw
        if isinstance(item, dict)
    ]


def _redact_data_dir(text: str, data_dir: str) -> str:
    resolved = str(Path(data_dir).resolve())
    return text.replace(resolved, "<data>").replace(str(data_dir), "<data>")


def _sanitize_public_markdown(markdown: str, data_dir: str, *, internal_hosts: tuple[str, ...] = ()) -> str:
    """Sanitize report markdown for the public route only.

    Admin responses return the stored markdown unchanged. Public reports must
    never reveal absolute filesystem paths, internal addresses, secrets, or
    budget figures.
    """
    # Line-level path redaction drops the remainder of any path-bearing line, so
    # spaced path components and UNC share tails cannot survive. Internal
    # addresses and secrets are still scrubbed token by token.
    text = redact_markdown_paths(_redact_data_dir(markdown, data_dir))
    text = redact_internal_addresses_and_secrets(text, internal_hosts=internal_hosts)
    return "\n".join(line for line in text.split("\n") if not line.startswith(_HIDDEN_LINE_PREFIXES))


async def _report(state, run_id: str, *, public: bool = False) -> dict[str, object]:
    snapshot = await state.reports.get(run_id)
    if snapshot is None:
        raise ApiError(404, "report_not_ready", "report is not ready")
    payload = snapshot.payload
    digest = snapshot.sha256
    markdown = snapshot.markdown
    if public:
        internal_hosts = _internal_hosts(state)
        payload = sanitize_public_payload(snapshot.payload, internal_hosts=internal_hosts)
        # Only rewrite keys the stored payload already has; injecting them would
        # change the shape and digest of legacy v2/v3 reports. Key presence (not
        # type) gates the whitelist so malformed non-mapping usage is neutralized.
        if "usage" in snapshot.payload:
            payload = {**payload, "usage": public_report_usage(payload)}
        if "truncated_sample_counts" in snapshot.payload:
            payload = {
                **payload,
                "truncated_sample_counts": public_truncated_sample_counts(payload),
            }
        # The stored digest covers the unsanitized payload, so recompute it with
        # the same canonicalization used by ReportRepository.upsert.
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        markdown = _sanitize_public_markdown(
            markdown,
            state.config.storage.data_dir,
            internal_hosts=internal_hosts,
        )
    return {
        "run_id": snapshot.run_id,
        "version": snapshot.version,
        "sha256": digest,
        "markdown": markdown,
        "report": payload,
    }


def _artifact_path(state, run_id: str, name: str) -> Path:
    """Resolve a stored artifact name to a file, refusing anything outside the run."""
    paths = run_paths(state.config.storage.data_dir, run_id)
    if _SAFE_NAME.fullmatch(name) is not None:
        base, relative = paths.root, name
    elif (match := _AIG_NAME.fullmatch(name)) is not None:
        base, relative = paths.aig_dir, f"{match.group(1)}.{match.group(2)}"
    else:
        raise ApiError(404, "not_found", "artifact not found")
    root = base.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(root):
        raise ApiError(404, "not_found", "artifact not found")
    return path


async def _artifact(state, run_id: str, name: str, *, public: bool) -> FileResponse:
    row = await state.artifacts.get(run_id, name)
    if row is None or (public and row.visibility != "public"):
        raise ApiError(404, "not_found", "artifact not found")
    path = _artifact_path(state, run_id, row.name)
    if not path.is_file():
        raise ApiError(404, "not_found", "artifact not found")
    return FileResponse(path, media_type=row.content_type, filename=Path(row.name).name)


@router.get("/runs/{run_id}/report")
async def admin_report(
    run_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
) -> dict[str, object]:
    state = get_state(request)
    await _load_run(state, run_id)
    return await _report(state, run_id)


@router.get("/runs/{run_id}/artifacts/{name:path}")
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
