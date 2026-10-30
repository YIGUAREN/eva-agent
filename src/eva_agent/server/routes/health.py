"""Liveness probe."""

from fastapi import APIRouter, Request
from sqlalchemy import text

from eva_agent.server.app import get_state
from eva_agent.server.errors import ApiError

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    state = get_state(request)
    try:
        async with state.session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        raise ApiError(503, "unavailable", "database unavailable") from None
    return {"status": "ok", "version": "4"}
