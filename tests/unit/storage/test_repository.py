import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind, PastedInput, RunInput, RunStatus
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import RunEventCounterRow, RunEventRow, RunRow
from eva_agent.storage.repository import RunRepository


def sample_input() -> RunInput:
    return RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q", response="a"))


def sample_budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=600.0, max_target_requests=10)


@pytest_asyncio.fixture
async def session_factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    db_path = (tmp_path / "test.db").as_posix()
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    await init_db(engine)
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def repo(session_factory: async_sessionmaker[AsyncSession]) -> RunRepository:
    return RunRepository(session_factory)


async def test_create_and_get(repo: RunRepository) -> None:
    await repo.create("run-1", sample_input(), sample_budget())

    row = await repo.get("run-1")

    assert row is not None
    assert row.status == RunStatus.QUEUED.value
    assert row.input_json["kind"] == "paste"
    assert row.budget_json["max_target_requests"] == 10
    assert row.is_public is False


async def test_create_initializes_event_counter(
    repo: RunRepository, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await repo.create("run-1", sample_input(), sample_budget())

    async with session_factory() as session:
        counter = await session.get(RunEventCounterRow, "run-1")

    assert counter is not None
    assert counter.next_seq == 0


async def test_get_missing_returns_none(repo: RunRepository) -> None:
    assert await repo.get("missing") is None


async def test_set_status(repo: RunRepository) -> None:
    await repo.create("run-1", sample_input(), sample_budget())

    await repo.set_status("run-1", RunStatus.RUNNING)

    row = await repo.get("run-1")
    assert row is not None
    assert row.status == RunStatus.RUNNING.value


async def test_set_status_unknown_run_raises(repo: RunRepository) -> None:
    with pytest.raises(KeyError):
        await repo.set_status("missing", RunStatus.RUNNING)


async def test_append_event_increments_seq(
    repo: RunRepository, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await repo.create("run-1", sample_input(), sample_budget())

    first = await repo.append_event("run-1", "run.status", {"status": "queued"})
    second = await repo.append_event("run-1", "run.status", {"status": "running"})
    third = await repo.append_event("run-1", "run.status", {"status": "completed"})

    assert [first, second, third] == [1, 2, 3]
    async with session_factory() as session:
        rows = (
            await session.scalars(select(RunEventRow).where(RunEventRow.run_id == "run-1").order_by(RunEventRow.seq))
        ).all()
    assert [row.seq for row in rows] == [1, 2, 3]


async def test_append_event_is_safe_under_concurrency(
    repo: RunRepository, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await repo.create("run-1", sample_input(), sample_budget())

    async def append_with_marker(marker: str) -> tuple[int, str]:
        seq = await repo.append_event("run-1", "run.status", {"marker": marker})
        return seq, marker

    results = await asyncio.gather(*(append_with_marker(f"marker-{i}") for i in range(8)))
    seqs = [seq for seq, _ in results]

    assert sorted(seqs) == list(range(1, 9))
    async with session_factory() as session:
        rows = (await session.scalars(select(RunEventRow).where(RunEventRow.run_id == "run-1"))).all()
    assert len(rows) == 8
    assert len({row.seq for row in rows}) == 8
    rows_by_seq = {row.seq: row for row in rows}
    for seq, marker in results:
        assert rows_by_seq[seq].payload_json == {"marker": marker}


async def test_append_event_unknown_run_raises_without_event(
    repo: RunRepository, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    with pytest.raises(KeyError):
        await repo.append_event("missing", "run.status")

    async with session_factory() as session:
        count = await session.scalar(
            select(func.count()).select_from(RunEventRow).where(RunEventRow.run_id == "missing")
        )
    assert count == 0


async def test_deleting_run_cascades_events_and_counter(
    repo: RunRepository, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await repo.create("run-1", sample_input(), sample_budget())
    await repo.append_event("run-1", "run.status")

    async with session_factory() as session:
        await session.execute(delete(RunRow).where(RunRow.id == "run-1"))
        await session.commit()

    async with session_factory() as session:
        event_count = await session.scalar(
            select(func.count()).select_from(RunEventRow).where(RunEventRow.run_id == "run-1")
        )
        counter = await session.get(RunEventCounterRow, "run-1")
    assert event_count == 0
    assert counter is None
