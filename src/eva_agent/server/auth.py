"""Server-side sessions and CSRF protection for the administrator."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import Depends, Request

from eva_agent.server.app import get_state
from eva_agent.server.errors import ApiError
from eva_agent.server.state import AppState
from eva_agent.storage.models import AdminSessionRow, ensure_utc

SESSION_COOKIE = "eva_session"
CSRF_HEADER = "X-CSRF-Token"
SLIDING_TOUCH_S = 60.0


@dataclass(frozen=True)
class IssuedSession:
    session_id: str
    csrf_token: str
    expires_at: datetime
    absolute_expires_at: datetime


def hash_session_id(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def verify_csrf(token: str, expected: str) -> bool:
    if not token or not expected:
        return False
    return secrets.compare_digest(token, expected)


async def issue_session(state: AppState) -> IssuedSession:
    if not state.admin_password_hash:
        raise RuntimeError("ADMIN_PASSWORD_HASH is not configured")
    now = state.now()
    web = state.config.web
    session_id = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    absolute = now + timedelta(seconds=web.session_absolute_ttl_s)
    expires = min(now + timedelta(seconds=web.session_ttl_s), absolute)
    await state.sessions.create(hash_session_id(session_id), csrf_token, expires, absolute)
    return IssuedSession(session_id, csrf_token, expires, absolute)


async def current_session(request: Request) -> AdminSessionRow:
    state = get_state(request)
    raw = request.cookies.get(SESSION_COOKIE)
    if not raw:
        raise ApiError(401, "unauthenticated", "login required")
    session = await state.sessions.get(hash_session_id(raw))
    if session is None:
        raise ApiError(401, "unauthenticated", "session expired")
    now = state.now()
    if (now - ensure_utc(session.last_seen_at)).total_seconds() >= SLIDING_TOUCH_S:
        expires = min(
            now + timedelta(seconds=state.config.web.session_ttl_s),
            ensure_utc(session.absolute_expires_at),
        )
        await state.sessions.touch(session.id_hash, expires_at=expires, now=now)
    return session


async def csrf_protected(
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
) -> AdminSessionRow:
    if not verify_csrf(request.headers.get(CSRF_HEADER, ""), session.csrf_token):
        raise ApiError(403, "csrf_failed", "missing or invalid CSRF token")
    return session
