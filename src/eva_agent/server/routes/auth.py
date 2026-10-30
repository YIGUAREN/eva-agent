"""Administrator authentication routes."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from eva_agent.server.app import get_state
from eva_agent.server.auth import SESSION_COOKIE, csrf_protected, current_session, issue_session
from eva_agent.server.errors import ApiError
from eva_agent.server.passwords import verify_password
from eva_agent.storage.models import AdminSessionRow, ensure_utc

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: str = Field(min_length=1, max_length=1_024)


def _client_key(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def _origin_parts(value: str) -> tuple[str, str, int] | None:
    """Normalize an origin/URL into ``(scheme, hostname, port)``."""
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if scheme not in {"http", "https"} or not hostname:
        return None
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, hostname.lower(), port


def _validate_login_origin(request: Request, public_base_url: str) -> None:
    """Reject cross-origin browser logins, while allowing non-browser clients.

    ``Origin`` wins when present; otherwise ``Referer`` is checked; when neither
    header is sent (CLI clients), the request is allowed.
    """
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    candidate = origin if origin is not None else referer
    if candidate is None:
        return
    expected = _origin_parts(public_base_url)
    if expected is None or _origin_parts(candidate) != expected:
        raise ApiError(403, "forbidden", "cross-origin login rejected")


@router.post("/login")
async def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, str]:
    state = get_state(request)
    _validate_login_origin(request, state.config.public_base_url)
    key = _client_key(request)
    if state.login_limiter.is_blocked(key):
        raise ApiError(429, "rate_limited", "too many failed login attempts")
    if not verify_password(payload.password, state.admin_password_hash):
        state.login_limiter.record_failure(key)
        raise ApiError(401, "invalid_credentials", "invalid credentials")
    state.login_limiter.reset(key)
    issued = await issue_session(state)
    response.set_cookie(
        SESSION_COOKIE,
        issued.session_id,
        max_age=state.config.web.session_absolute_ttl_s,
        httponly=True,
        samesite="lax",
        secure=state.config.web.cookie_secure,
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return {"csrf_token": issued.csrf_token, "expires_at": issued.expires_at.isoformat()}


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    response: Response,
    session: Annotated[AdminSessionRow, Depends(csrf_protected)],
) -> None:
    state = get_state(request)
    await state.sessions.revoke(session.id_hash)
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        httponly=True,
        samesite="lax",
        secure=state.config.web.cookie_secure,
    )


@router.get("/me")
async def me(
    session: Annotated[AdminSessionRow, Depends(current_session)],
    response: Response,
) -> dict[str, str]:
    response.headers["Cache-Control"] = "no-store"
    return {
        "csrf_token": session.csrf_token,
        "expires_at": ensure_utc(session.expires_at).isoformat(),
    }
