"""Server-sent events replayed from persisted run events."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse

from eva_agent.domain.plan import Dimension
from eva_agent.domain.run import RunStatus
from eva_agent.server.app import get_state
from eva_agent.server.auth import current_session
from eva_agent.server.routes.runs import _internal_hosts, _load_public_run, _load_run
from eva_agent.server.schemas import sanitize_public_payload
from eva_agent.server.state import AppState
from eva_agent.storage.models import AdminSessionRow

router = APIRouter(prefix="/api", tags=["events"])

TERMINAL = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})
HEARTBEAT_S = 15.0

_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# Budget money keys that must never reach the public event stream. This is the
# fallback for event types that are not explicitly allowlisted below.
_PUBLIC_BUDGET_MONEY_KEYS = frozenset(
    {
        "spent_usd",
        "remaining_usd",
        "spent",
        "remaining",
        "finalize_estimate",
        "finalize_estimate_usd",
    }
)

# Public budget events expose an explicit, type-checked shape instead of
# denylisting names: ``budget.exhausted`` carries spent_usd/remaining_usd and
# ``budget.gate`` carries spent/remaining/finalize_estimate. Allowlisting keys
# is immune to casing/suffix variants and nested money objects, and checking
# the surviving values means a hostile object cannot hide behind an allowed
# key. Admin streams keep the full payloads unchanged.
_PUBLIC_BUDGET_EVENT_TYPES = frozenset({"budget.exhausted", "budget.gate"})

# Derived from the enum so a future dimension is accepted automatically.
_PUBLIC_DIMENSION_NAMES = frozenset(item.value for item in Dimension)


def _without_budget_money(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_budget_money(item)
            for key, item in value.items()
            if key not in _PUBLIC_BUDGET_MONEY_KEYS
        }
    if isinstance(value, list):
        return [_without_budget_money(item) for item in value]
    return value


def _public_event_payload(event_type: str, payload: object) -> object:
    """Reduce public budget events to their allowlisted, type-checked shape.

    ``budget.exhausted`` exposes ``dimension`` only when it is a real
    ``Dimension`` name; ``budget.gate`` exposes ``truncated_dimensions`` only
    when it is a list, filtered down to its real dimension names. Any malformed
    budget payload or value becomes an empty object. Other event types keep the
    generic money-key denylist.
    """
    if event_type not in _PUBLIC_BUDGET_EVENT_TYPES:
        return _without_budget_money(payload)
    if not isinstance(payload, dict):
        return {}
    if event_type == "budget.exhausted":
        dimension = payload.get("dimension")
        if isinstance(dimension, str) and dimension in _PUBLIC_DIMENSION_NAMES:
            return {"dimension": dimension}
        return {}
    truncated = payload.get("truncated_dimensions")
    if not isinstance(truncated, list):
        return {}
    return {
        "truncated_dimensions": [
            item
            for item in truncated
            if isinstance(item, str) and item in _PUBLIC_DIMENSION_NAMES
        ]
    }


def _cursor(raw: str | None) -> int:
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _format(seq: int, event_type: str, payload: object) -> str:
    body = json.dumps({"seq": seq, "type": event_type, "payload": payload}, ensure_ascii=False)
    return f"id: {seq}\ndata: {body}\n\n"


async def _stream(
    state: AppState,
    run_id: str,
    cursor: int,
    *,
    public: bool,
    internal_hosts: tuple[str, ...] = (),
) -> AsyncIterator[str]:
    elapsed = 0.0
    while True:
        events = await state.runs.list_events(run_id, after_seq=cursor)
        for event in events:
            cursor = event.seq
            if public:
                payload = _public_event_payload(
                    event.type,
                    sanitize_public_payload(event.payload_json, internal_hosts=internal_hosts),
                )
            else:
                payload = event.payload_json
            yield _format(event.seq, event.type, payload)
        if events:
            elapsed = 0.0
            continue
        row = await state.runs.get(run_id)
        if row is None or RunStatus(row.status) in TERMINAL:
            return
        poll = max(0.05, state.config.web.poll_interval_s)
        await asyncio.sleep(poll)
        elapsed += poll
        if elapsed >= HEARTBEAT_S:
            elapsed = 0.0
            yield ": heartbeat\n\n"


@router.get("/runs/{run_id}/events")
async def admin_events(
    run_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    state = get_state(request)
    await _load_run(state, run_id)
    return StreamingResponse(
        _stream(state, run_id, _cursor(last_event_id), public=False),
        media_type="text/event-stream",
        headers=_HEADERS,
    )


@router.get("/public/runs/{run_id}/events")
async def public_events(
    run_id: str,
    request: Request,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    state = get_state(request)
    await _load_public_run(state, run_id)
    return StreamingResponse(
        _stream(
            state,
            run_id,
            _cursor(last_event_id),
            public=True,
            internal_hosts=_internal_hosts(state),
        ),
        media_type="text/event-stream",
        headers=_HEADERS,
    )
