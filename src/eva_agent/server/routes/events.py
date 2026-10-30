"""Server-sent events replayed from persisted run events."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse

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
            payload = (
                sanitize_public_payload(event.payload_json, internal_hosts=internal_hosts)
                if public
                else event.payload_json
            )
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
