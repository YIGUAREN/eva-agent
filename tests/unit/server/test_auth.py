from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from eva_agent.server.auth import (
    SESSION_COOKIE,
    SLIDING_TOUCH_S,
    csrf_protected,
    current_session,
    hash_session_id,
    issue_session,
    verify_csrf,
)
from eva_agent.server.errors import ApiError
from eva_agent.server.ratelimit import LoginRateLimiter
from eva_agent.storage.models import ensure_utc
from tests.support.server import build_test_app


def _request(app: FastAPI, session_id: str, *, csrf_token: str | None = None) -> Request:
    headers = [(b"cookie", f"{SESSION_COOKIE}={session_id}".encode("latin-1"))]
    if csrf_token is not None:
        headers.append((b"x-csrf-token", csrf_token.encode("utf-8")))
    return Request({"type": "http", "app": app, "headers": headers})


async def test_issue_session_stores_only_a_hash(tmp_path) -> None:
    _, state = await build_test_app(tmp_path)

    issued = await issue_session(state)

    assert issued.session_id != issued.csrf_token
    assert await state.sessions.get(issued.session_id) is None
    stored = await state.sessions.get(hash_session_id(issued.session_id))
    assert stored is not None
    assert stored.csrf_token == issued.csrf_token
    await state.engine.dispose()


async def test_expired_session_is_not_returned(tmp_path) -> None:
    _, state = await build_test_app(tmp_path)
    issued = await issue_session(state)
    moment = datetime.now(UTC) + timedelta(seconds=state.config.web.session_absolute_ttl_s + 1)

    assert await state.sessions.get(hash_session_id(issued.session_id), now=moment) is None
    await state.engine.dispose()


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value


def test_rate_limiter_blocks_after_five_failures() -> None:
    clock = FakeClock()
    limiter = LoginRateLimiter(clock=clock)

    for _ in range(5):
        limiter.record_failure("127.0.0.1")

    assert limiter.is_blocked("127.0.0.1") is True
    assert limiter.is_blocked("10.0.0.1") is False


def test_rate_limiter_window_expires_and_reset_clears() -> None:
    clock = FakeClock()
    limiter = LoginRateLimiter(clock=clock)
    for _ in range(5):
        limiter.record_failure("127.0.0.1")

    clock.value += 901
    assert limiter.is_blocked("127.0.0.1") is False

    for _ in range(5):
        limiter.record_failure("127.0.0.1")
    assert limiter.is_blocked("127.0.0.1") is True

    limiter.reset("127.0.0.1")
    assert limiter.is_blocked("127.0.0.1") is False


def test_verify_csrf_uses_constant_time_comparison() -> None:
    assert verify_csrf("abc", "abc") is True
    assert verify_csrf("abc", "abd") is False
    assert verify_csrf("", "abc") is False
    assert verify_csrf("abc", "") is False
    assert verify_csrf("令牌", "abc") is False


async def test_non_ascii_csrf_token_is_rejected_with_403(tmp_path) -> None:
    app, state = await build_test_app(tmp_path)
    issued = await issue_session(state)
    request = _request(app, issued.session_id, csrf_token="令牌")

    session = await current_session(request)
    with pytest.raises(ApiError) as info:
        await csrf_protected(request, session)

    assert info.value.status_code == 403
    await state.engine.dispose()


async def test_current_session_uses_injected_clock_for_expiry(tmp_path) -> None:
    app, state = await build_test_app(tmp_path)
    issued = await issue_session(state)
    later = state.now() + timedelta(seconds=state.config.web.session_ttl_s + 1)
    state.now = lambda: later

    with pytest.raises(ApiError) as info:
        await current_session(_request(app, issued.session_id))

    assert info.value.status_code == 401
    await state.engine.dispose()


async def test_current_session_returns_post_touch_expiry(tmp_path) -> None:
    app, state = await build_test_app(tmp_path)
    issued = await issue_session(state)
    later = state.now() + timedelta(seconds=SLIDING_TOUCH_S + 5)
    state.now = lambda: later

    session = await current_session(_request(app, issued.session_id))

    expected = min(later + timedelta(seconds=state.config.web.session_ttl_s), issued.absolute_expires_at)
    assert ensure_utc(session.expires_at) == expected
    assert ensure_utc(session.last_seen_at) == later
    await state.engine.dispose()
