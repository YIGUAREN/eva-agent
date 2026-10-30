from datetime import UTC, datetime, timedelta

from eva_agent.server.auth import hash_session_id, issue_session, verify_csrf
from eva_agent.server.ratelimit import LoginRateLimiter
from tests.support.server import build_test_app


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

    limiter.record_failure("127.0.0.1")
    limiter.reset("127.0.0.1")
    assert limiter.is_blocked("127.0.0.1") is False


def test_verify_csrf_uses_constant_time_comparison() -> None:
    assert verify_csrf("abc", "abc") is True
    assert verify_csrf("abc", "abd") is False
    assert verify_csrf("", "abc") is False
    assert verify_csrf("abc", "") is False
