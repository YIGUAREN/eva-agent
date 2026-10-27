from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind, PastedInput, RunInput, RunStatus
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.repository import RunRepository


def sample_input() -> RunInput:
    return RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q", response="a"))


def sample_budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=600.0, max_target_requests=10)


@pytest_asyncio.fixture
async def repo(tmp_path: Path) -> AsyncIterator[RunRepository]:
    db_path = (tmp_path / "test.db").as_posix()
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    await init_db(engine)
    try:
        yield RunRepository(create_session_factory(engine))
    finally:
        await engine.dispose()


async def test_create_and_get(repo: RunRepository) -> None:
    await repo.create("run-1", sample_input(), sample_budget())

    row = await repo.get("run-1")

    assert row is not None
    assert row.status == RunStatus.QUEUED.value
    assert row.input_json["kind"] == "paste"
    assert row.budget_json["max_target_requests"] == 10
    assert row.is_public is False


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


async def test_append_event_increments_seq(repo: RunRepository) -> None:
    await repo.create("run-1", sample_input(), sample_budget())

    first = await repo.append_event("run-1", "run.status", {"status": "queued"})
    second = await repo.append_event("run-1", "run.status", {"status": "running"})

    assert first == 1
    assert second == 2
