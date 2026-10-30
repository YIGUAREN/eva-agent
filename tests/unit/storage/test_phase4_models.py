from datetime import UTC, datetime, timedelta
from pathlib import Path

from eva_agent.domain.budget import Budget
from eva_agent.domain.run import RunStatus
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.reports import ReportRepository
from eva_agent.storage.repository import RunRepository
from eva_agent.storage.sessions import AdminSessionRepository
from tests.support.runtime import paste_input


def _budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1)


async def test_run_round_trip_and_public_listing(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'eva.db').as_posix()}")
    await init_db(engine)
    runs = RunRepository(create_session_factory(engine))

    await runs.create("run_a", paste_input(), _budget(), idempotency_key="key-1", created_by="admin")
    await runs.create("run_b", paste_input(), _budget())
    await runs.set_status("run_b", RunStatus.COMPLETED)
    await runs.set_public("run_b", True)

    assert (await runs.get_by_idempotency_key("key-1")).id == "run_a"  # type: ignore[union-attr]
    public = await runs.list_public(cursor=None, limit=10)
    assert [row.id for row in public] == ["run_b"]
    await engine.dispose()


async def test_events_are_ordered_and_resumable(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'eva.db').as_posix()}")
    await init_db(engine)
    runs = RunRepository(create_session_factory(engine))
    await runs.create("run_a", paste_input(), _budget())

    assert await runs.append_event("run_a", "one", {}) == 1
    assert await runs.append_event("run_a", "two", {"x": 1}) == 2

    assert [row.type for row in await runs.list_events("run_a", after_seq=1)] == ["two"]
    assert await runs.latest_event_seq("run_a") == 2
    await engine.dispose()


async def test_reports_and_sessions(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'eva.db').as_posix()}")
    await init_db(engine)
    factory = create_session_factory(engine)
    runs = RunRepository(factory)
    reports = ReportRepository(factory)
    sessions = AdminSessionRepository(factory)
    await runs.create("run_a", paste_input(), _budget())

    await reports.upsert("run_a", 3, {"status": "completed"}, "# report")
    snapshot = await reports.get("run_a")
    assert snapshot is not None and snapshot.markdown == "# report"

    later = datetime.now(UTC) + timedelta(hours=1)
    await sessions.create("hash", "csrf", later, later + timedelta(hours=1))
    assert (await sessions.get("hash")) is not None
    await sessions.revoke("hash")
    assert (await sessions.get("hash")) is None
    await engine.dispose()
