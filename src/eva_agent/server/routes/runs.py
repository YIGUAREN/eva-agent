"""Run lifecycle routes."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.exc import IntegrityError

from eva_agent.adapters.openjudge.catalog import CatalogError, load_catalog
from eva_agent.config.settings import ConfigError
from eva_agent.config.targets import load_targets
from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.run import GraderSelection, InputKind, RunStatus
from eva_agent.runtime.budget_decisions import (
    BudgetOverrunRequired,
    InvalidBudgetExtension,
    ensure_summarize_allowed,
    validate_extension,
)
from eva_agent.server.app import get_state
from eva_agent.server.auth import csrf_protected, current_session
from eva_agent.server.errors import ApiError
from eva_agent.server.schemas import (
    ApproveRequest,
    BudgetDecisionRequest,
    CreateRunRequest,
    PublishRequest,
    decode_cursor,
    encode_cursor,
    public_run_detail,
    run_detail,
    run_summary,
)
from eva_agent.server.state import AppState
from eva_agent.storage.commands import ActiveCommandExists
from eva_agent.storage.models import AdminSessionRow, RunRow, RunUsageRow
from eva_agent.storage.uploads import UPLOAD_EXTRACTED

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


async def _audit_rejection(state: AppState, row: RunRow, action: str) -> None:
    """Record a rejected privileged action before its error response is raised."""
    await state.audit.record(
        AuditEvent(
            run_id=row.id,
            actor="admin",
            action=action,
            outcome="rejected",
            authorization_ref=row.authorization_ref,
        )
    )


def _usage_view(row: RunUsageRow | None) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "spent_usd": float(row.spent_usd),
        "reserved_usd": float(row.reserved_usd),
        "spent_wall_clock_s": float(row.spent_wall_clock_s or 0.0),
        "target_requests": int(row.target_requests or 0),
        "aig_requests": int(row.aig_requests or 0),
    }


def _internal_hosts(state: AppState) -> tuple[str, ...]:
    """Return the AIG host that public serializers must redact, when configured."""
    aig = state.config.aig
    if aig is None:
        return ()
    try:
        hostname = urlsplit(aig.base_url).hostname
    except ValueError:
        return ()
    return (hostname,) if hostname else ()


def _validate_grader_selection(state: AppState, selection: GraderSelection | None) -> None:
    """Reject selections that name no grader, an unknown id, or a wrong-dimension id."""
    if selection is None:
        return
    if selection.is_empty():
        raise ApiError(422, "no_graders", "至少勾选一个评分器")
    try:
        catalog = load_catalog(Path(state.config.openjudge.catalog_path))
    except CatalogError:
        raise ApiError(503, "unavailable", "grader catalog unavailable") from None
    groups = (("quality", selection.quality), ("security", selection.security))
    for dimension, grader_ids in groups:
        for grader_id in grader_ids:
            entry = catalog.get(grader_id)
            if entry is None:
                raise ApiError(422, "unknown_grader", f"未知评分器：{grader_id}")
            if dimension not in entry.tags:
                raise ApiError(422, "invalid_grader", f"评分器 {grader_id} 不属于 {dimension} 维度")


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
    _validate_grader_selection(state, payload.input.graders)
    if payload.idempotency_key:
        existing = await state.runs.get_by_idempotency_key(payload.idempotency_key)
        if existing is not None:
            response.status_code = 200
            return {"run_id": existing.id, "status": existing.status}

    if payload.input.kind is InputKind.UPLOAD:
        upload_id = payload.input.upload_id or ""
        upload = await state.uploads.get(upload_id)
        if upload is None:
            raise ApiError(422, "unknown_upload", "上传记录不存在")
        if upload.status != UPLOAD_EXTRACTED:
            raise ApiError(409, "upload_not_ready", "上传包尚未解压完成")

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
    state = get_state(request)
    row = await _load_run(state, run_id)
    gate = await state.decisions.latest_gate(run_id)
    return run_detail(
        row,
        usage=_usage_view(await state.usage.get(run_id)),
        budget_gate=dict(gate.payload_json) if gate is not None else None,
    )


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
    internal_hosts = _internal_hosts(state)
    return {
        "items": [public_run_detail(row, internal_hosts=internal_hosts) for row in rows],
        "next_cursor": next_cursor,
    }


@router.get("/public/runs/{run_id}")
async def get_public_run(run_id: str, request: Request) -> dict[str, object]:
    state = get_state(request)
    row = await _load_public_run(state, run_id)
    return public_run_detail(row, internal_hosts=_internal_hosts(state))


@router.post("/runs/{run_id}/approve", status_code=202)
async def approve_run(
    run_id: str,
    payload: ApproveRequest,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(csrf_protected)],
) -> dict[str, str]:
    state = get_state(request)
    row = await _load_run(state, run_id)
    action = "run.approve" if payload.approved else "run.reject"
    if row.status != RunStatus.WAITING_APPROVAL.value:
        await _audit_rejection(state, row, action)
        raise ApiError(409, "invalid_state", "run is not waiting for approval")
    try:
        await state.commands.enqueue(
            run_id,
            "approve",
            {"approved": payload.approved, "plan": payload.plan},
        )
    except ActiveCommandExists:
        await _audit_rejection(state, row, action)
        raise ApiError(409, "command_pending", "another command is already pending") from None
    await state.audit.record(
        AuditEvent(
            run_id=run_id,
            actor="admin",
            action=action,
            outcome="accepted",
            authorization_ref=row.authorization_ref,
        )
    )
    return {"run_id": run_id, "status": row.status, "queued": "approve"}


@router.post("/runs/{run_id}/budget/decision", status_code=202)
async def decide_run_budget(
    run_id: str,
    payload: BudgetDecisionRequest,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(csrf_protected)],
) -> dict[str, str]:
    state = get_state(request)
    row = await _load_run(state, run_id)
    action = f"run.budget.{payload.action}"
    # A terminal run cannot be decided again; report the current state.
    if row.status in {
        RunStatus.COMPLETED.value,
        RunStatus.FAILED.value,
        RunStatus.CANCELLED.value,
    }:
        return {"run_id": run_id, "status": row.status, "queued": ""}
    recorded = await state.decisions.latest_decision(run_id)
    gate = await state.decisions.latest_gate(run_id)
    if recorded is not None and gate is None:
        # The latest gate already carries a decision, so this submission must
        # not decide again.  A RUNNING row means the resume finished in the
        # graph but the row was never synced: queue a recovery so the worker
        # converges from the checkpoint and runs post-processing exactly once.
        recovery = dict(recorded.payload_json.get("decision") or {})
        if row.status == RunStatus.RUNNING.value and recovery:
            try:
                await state.commands.enqueue(run_id, "budget_decision", recovery)
            except ActiveCommandExists:
                await _audit_rejection(state, row, action)
                raise ApiError(409, "command_pending", "another command is already pending") from None
            # The submission did not decide anything: the first decision for the
            # seq is final and the recorded one is what actually runs.  Record
            # that replayed decision rather than the submitted action.
            replayed = str(recovery.get("action") or "")
            await state.audit.record(
                AuditEvent(
                    run_id=run_id,
                    actor="admin",
                    action=f"run.budget.{replayed}" if replayed else action,
                    outcome="replayed",
                    authorization_ref=row.authorization_ref,
                )
            )
            return {"run_id": run_id, "status": row.status, "queued": "budget_decision"}
        return {"run_id": run_id, "status": row.status, "queued": ""}
    if gate is None or row.status not in {
        RunStatus.WAITING_BUDGET.value,
        # A RUNNING row with a pending gate is the crash window between the gate
        # checkpoint and _sync; the worker reconciles it when deciding.
        RunStatus.RUNNING.value,
    }:
        await _audit_rejection(state, row, action)
        raise ApiError(409, "invalid_state", "run is not waiting for a budget decision")
    gate_payload = dict(gate.payload_json)
    finalize_estimate = Decimal(str(gate_payload["finalize_estimate"]))
    remaining = Decimal(str(gate_payload["remaining"]))
    usage = state.usage.to_usage(await state.usage.get(run_id))

    command_payload: dict[str, object]
    if payload.action == "summarize":
        try:
            ensure_summarize_allowed(
                finalize_estimate=finalize_estimate,
                remaining=remaining,
                confirm_overrun=payload.confirm_overrun,
            )
        except BudgetOverrunRequired as exc:
            await _audit_rejection(state, row, action)
            raise ApiError(
                409,
                "budget_overrun",
                "预算不足以完成收尾，请确认超支后重试",
                details={
                    "finalize_estimate": float(exc.finalize_estimate),
                    "remaining": float(exc.remaining),
                },
            ) from None
        command_payload = {
            "action": "summarize",
            "confirm_overrun": payload.confirm_overrun,
        }
    else:
        try:
            validate_extension(payload.budget, usage)
        except InvalidBudgetExtension as exc:
            await _audit_rejection(state, row, action)
            raise ApiError(
                400,
                "invalid_request",
                str(exc),
                details={"field": exc.field},
            ) from None
        command_payload = {"action": "extend", "budget": payload.budget.model_dump(mode="json")}

    try:
        await state.commands.enqueue(run_id, "budget_decision", command_payload)
    except ActiveCommandExists:
        await _audit_rejection(state, row, action)
        raise ApiError(409, "command_pending", "another command is already pending") from None
    await state.audit.record(
        AuditEvent(
            run_id=run_id,
            actor="admin",
            action=action,
            outcome="accepted",
            authorization_ref=row.authorization_ref,
        )
    )
    return {"run_id": run_id, "status": row.status, "queued": "budget_decision"}


@router.post("/runs/{run_id}/cancel", status_code=202)
async def cancel_run(
    run_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(csrf_protected)],
) -> dict[str, str]:
    state = get_state(request)
    row = await _load_run(state, run_id)
    if row.status not in {
        RunStatus.QUEUED.value,
        RunStatus.RUNNING.value,
        RunStatus.WAITING_APPROVAL.value,
        RunStatus.WAITING_BUDGET.value,
    }:
        await _audit_rejection(state, row, "run.cancel")
        raise ApiError(409, "invalid_state", "run cannot be cancelled in its current state")
    try:
        await state.commands.enqueue(run_id, "cancel")
    except ActiveCommandExists:
        await _audit_rejection(state, row, "run.cancel")
        raise ApiError(409, "command_pending", "another command is already pending") from None
    await state.audit.record(
        AuditEvent(
            run_id=run_id,
            actor="admin",
            action="run.cancel",
            outcome="accepted",
            authorization_ref=row.authorization_ref,
        )
    )
    return {"run_id": run_id, "status": row.status, "queued": "cancel"}
