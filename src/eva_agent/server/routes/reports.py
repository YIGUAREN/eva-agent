"""Reports, downloadable artifacts, and public catalogs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse

from eva_agent.adapters.openjudge.catalog import CatalogError, load_catalog, summarize
from eva_agent.config.settings import ConfigError
from eva_agent.config.targets import load_targets
from eva_agent.domain.plan import Dimension
from eva_agent.runtime.artifacts import artifact_file
from eva_agent.server.app import get_state
from eva_agent.server.auth import current_session
from eva_agent.server.errors import ApiError
from eva_agent.server.routes.runs import _internal_hosts, _load_public_run, _load_run
from eva_agent.server.schemas import sanitize_public_markdown, sanitize_public_payload
from eva_agent.storage.models import AdminSessionRow

router = APIRouter(prefix="/api", tags=["reports"])

_PUBLIC_USAGE_KEYS = ("estimated", "cached", "fresh", "budget_enforced", "spent_usd")
_PUBLIC_USAGE_BOOL_KEYS = frozenset({"estimated", "budget_enforced"})
_PUBLIC_USAGE_MONEY_KEYS = frozenset({"spent_usd"})

_PUBLIC_SAMPLE_COUNT_KEYS = ("sample_count", "completed_samples", "missing_samples")

# Derived from the enum so a future dimension is accepted automatically.
_PUBLIC_DIMENSION_NAMES = frozenset(item.value for item in Dimension)


def _valid_usage_value(key: str, value: object) -> bool:
    """A whitelisted usage key is only public when its value has the right type."""
    if key in _PUBLIC_USAGE_BOOL_KEYS:
        return isinstance(value, bool)
    if key in _PUBLIC_USAGE_MONEY_KEYS:
        # bool is an int subclass and a hostile producer may send NaN/inf/negatives.
        # A huge int such as ``10**400`` raises OverflowError when converted to a
        # float, so it cannot be a finite JSON number either: drop it rather than
        # letting the conversion escape as a 500.
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            if value < 0:
                return False
            try:
                return math.isfinite(float(value))
            except OverflowError:
                return False
        if isinstance(value, float):
            return math.isfinite(value) and value >= 0
        return False
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def public_report_usage(payload: dict[str, object]) -> dict[str, object]:
    """Copy only whitelisted usage keys whose values are safe primitives.

    The actual spend (``spent_usd``) is deliberately public now; the budget cap
    is not. A hostile producer can put spend figures into a whitelisted key
    (for example ``{"cached": {"spent_usd": 12.3}}``), so key presence alone is
    not enough: each value is type-checked and dropped when it does not match
    its expected type. Money values must be non-negative finite numbers, so a
    bool, string, NaN, infinity, or negative value is dropped.
    """
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return {}
    return {
        key: raw[key]
        for key in _PUBLIC_USAGE_KEYS
        if key in raw and _valid_usage_value(key, raw[key])
    }


def public_truncated_dimensions(payload: dict[str, object]) -> list[str]:
    """Keep only real dimension names, never free text such as a spend figure.

    A non-list value becomes ``[]``; items that are not ``str`` or are not a
    ``Dimension`` member are dropped.
    """
    raw = payload.get("truncated_dimensions")
    if not isinstance(raw, list):
        return []
    return [
        item
        for item in raw
        if isinstance(item, str) and item in _PUBLIC_DIMENSION_NAMES
    ]


def _valid_sample_count(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def public_truncated_sample_counts(payload: dict[str, object]) -> list[dict[str, object]]:
    """Whitelist each truncation entry so hostile keys and values never go public.

    An entry survives only when its dimension is a real ``Dimension`` name and
    every supplied count is ``None`` or a non-negative integer (booleans
    excluded); otherwise the whole entry is dropped rather than partially
    copied.
    """
    raw = payload.get("truncated_sample_counts")
    if not isinstance(raw, list):
        return []
    public: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        dimension = item.get("dimension")
        if not isinstance(dimension, str) or dimension not in _PUBLIC_DIMENSION_NAMES:
            continue
        counts = {key: item[key] for key in _PUBLIC_SAMPLE_COUNT_KEYS if key in item}
        if not all(_valid_sample_count(value) for value in counts.values()):
            continue
        public.append({"dimension": dimension, **counts})
    return public


async def _report(state, run_id: str, *, public: bool = False) -> dict[str, object]:
    snapshot = await state.reports.get(run_id)
    if snapshot is None:
        raise ApiError(404, "report_not_ready", "report is not ready")
    # A malformed stored payload (a string or list) must not crash the public
    # rewrite gates; only mappings can carry the whitelisted keys. The admin
    # route still returns the stored value unchanged.
    stored_payload = snapshot.payload if isinstance(snapshot.payload, dict) else {}
    payload = snapshot.payload
    digest = snapshot.sha256
    markdown = snapshot.markdown
    if public:
        internal_hosts = _internal_hosts(state)
        payload = sanitize_public_payload(snapshot.payload, internal_hosts=internal_hosts)
        # Only rewrite keys the stored payload already has; injecting them would
        # change the shape and digest of legacy v2/v3 reports. Key presence (not
        # type) gates the whitelist so malformed non-mapping usage is neutralized.
        if "usage" in stored_payload:
            payload = {**payload, "usage": public_report_usage(stored_payload)}
        if "truncated_dimensions" in stored_payload:
            payload = {
                **payload,
                "truncated_dimensions": public_truncated_dimensions(stored_payload),
            }
        if "truncated_sample_counts" in stored_payload:
            payload = {
                **payload,
                "truncated_sample_counts": public_truncated_sample_counts(stored_payload),
            }
        # The stored digest covers the unsanitized payload, so recompute it with
        # the same canonicalization used by ReportRepository.upsert.
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        markdown = sanitize_public_markdown(
            markdown,
            data_dir=state.config.storage.data_dir,
            internal_hosts=internal_hosts,
        )
    return {
        "run_id": snapshot.run_id,
        "version": snapshot.version,
        "sha256": digest,
        "markdown": markdown,
        "report": payload,
    }


async def _artifact(state, run_id: str, name: str, *, public: bool) -> FileResponse:
    row = await state.artifacts.get(run_id, name)
    if row is None or (public and row.visibility != "public"):
        raise ApiError(404, "not_found", "artifact not found")
    try:
        path = artifact_file(state.config.storage.data_dir, run_id, row.name)
    except ValueError:
        raise ApiError(404, "not_found", "artifact not found") from None
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
