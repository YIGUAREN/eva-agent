"""Run lifecycle routes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.exc import IntegrityError

from eva_agent.config.settings import ConfigError
from eva_agent.config.targets import load_targets
from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.server.app import get_state
from eva_agent.server.auth import csrf_protected, current_session
from eva_agent.server.errors import ApiError
from eva_agent.server.schemas import (
    CreateRunRequest,
    PublishRequest,
    decode_cursor,
    encode_cursor,
    public_run_detail,
    run_detail,
    run_summary,
)
from eva_agent.server.state import AppState
from eva_agent.storage.models import AdminSessionRow, RunRow

router = APIRouter(prefix="/api", tags=["runs"])


async def _load_run(state: AppState, run_id: str) -> RunRow:
    row = await state.runs.get(run_id)
    if row is None:
        raise ApiError(404, "not_found", "run not found")
    return row


async def _load_public_run(state: AppState, run_id: str) -> RunRow:
    row = await _load_run(state, run_id)
    if not row.is_public or row.status != RunStatus.COMPLETED.value:
        raise ApiError(404, "not_found", "run not found")
    return row


def _target_names(state: AppState) -> Mapping[str, object]:
    try:
        return load_targets(Path(state.config.target.targets_path))
    except ConfigError:
        raise ApiError(503, "unavailable", "target catalog unavailable") from None


@router.post("/runs", status_code=201)
async def create_run(
    payload: CreateRunRequest,
    request: Request,
    response: Response,
    session: Annotated[AdminSessionRow, Depends(csrf_protected)],
) -> dict[str, str]:
    state = get_state(request)
    if payload.input.kind is InputKind.TARGET and payload.input.target_ref not in _target_names(state):
        raise ApiError(422, "unknown_target", "unknown target")
    if payload.idempotency_key:
        existing = await state.runs.get_by_idempotency_key(payload.idempotency_key)
        if existing is not None:
            response.status_code = 200
            return {"run_id": existing.id, "status": existing.status}

    run_id = f"run_{uuid4().hex}"
    try:
        await state.runs.create(
            run_id,
            payload.input,
            payload.budget,
            idempotency_key=payload.idempotency_key,
            authorization_ref=payload.authorization_ref,
            created_by="admin",
        )
    except IntegrityError:
        if payload.idempotency_key:
            existing = await state.runs.get_by_idempotency_key(payload.idempotency_key)
            if existing is not None:
                response.status_code = 200
                return {"run_id": existing.id, "status": existing.status}
        raise ApiError(409, "conflict", "run could not be created") from None

    await state.runs.append_event(run_id, "run.created", {"kind": payload.input.kind.value})
    await state.commands.enqueue(run_id, "start")
    return {"run_id": run_id, "status": RunStatus.QUEUED.value}


@router.get("/runs")
async def list_runs(
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, list[dict[str, object]]]:
    state = get_state(request)
    rows = await state.runs.list_recent(limit)
    return {"items": [run_summary(row) for row in rows]}


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
) -> dict[str, object]:
    return run_detail(await _load_run(get_state(request), run_id))


@router.post("/runs/{run_id}/publish")
async def publish_run(
    run_id: str,
    payload: PublishRequest,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(csrf_protected)],
) -> dict[str, object]:
    state = get_state(request)
    row = await _load_run(state, run_id)
    if payload.published and row.status != RunStatus.COMPLETED.value:
        raise ApiError(409, "not_completed", "only completed runs can be published")
    updated = await state.runs.set_public(run_id, payload.published, now=state.now())
    await state.audit.record(
        AuditEvent(
            run_id=run_id,
            actor="admin",
            action="run.publish" if payload.published else "run.unpublish",
            outcome="success",
            authorization_ref=row.authorization_ref,
        )
    )
    return run_detail(updated)


@router.get("/public/runs")
async def list_public_runs(
    request: Request,
    cursor: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, object]:
    state = get_state(request)
    rows = await state.runs.list_public(cursor=decode_cursor(cursor) if cursor else None, limit=limit)
    next_cursor = encode_cursor(rows[-1]) if len(rows) == limit else None
    return {"items": [public_run_detail(row) for row in rows], "next_cursor": next_cursor}


@router.get("/public/runs/{run_id}")
async def get_public_run(run_id: str, request: Request) -> dict[str, object]:
    return public_run_detail(await _load_public_run(get_state(request), run_id))
