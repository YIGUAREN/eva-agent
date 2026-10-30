from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from eva_agent.domain.budget import Budget
from eva_agent.storage.commands import ActiveCommandExists, RunCommandRepository
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import paste_input


def _budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1)


async def _repository(tmp_path: Path) -> tuple[RunCommandRepository, RunRepository, AsyncEngine]:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'eva.db').as_posix()}")
    await init_db(engine)
    factory = create_session_factory(engine)
    runs = RunRepository(factory)
    await runs.create("run_a", paste_input(), _budget())
    return RunCommandRepository(factory), runs, engine


async def test_only_one_active_non_cancel_command(tmp_path: Path) -> None:
    commands, _, engine = await _repository(tmp_path)
    await commands.enqueue("run_a", "start")

    with pytest.raises(ActiveCommandExists):
        await commands.enqueue("run_a", "approve", {"approved": True})
    await commands.enqueue("run_a", "cancel")
    await engine.dispose()


async def test_claim_prioritizes_cancel_and_resets_stale(tmp_path: Path) -> None:
    commands, _, engine = await _repository(tmp_path)
    await commands.enqueue("run_a", "start")
    claimed = await commands.claim_next()
    assert claimed is not None and claimed.kind == "start"

    await commands.finish(claimed.id, "done")
    await commands.enqueue("run_a", "approve", {"approved": True})
    await commands.enqueue("run_a", "cancel")

    first = await commands.claim_next()
    assert first is not None and first.kind == "cancel"
    await commands.finish(first.id, "done")

    stale = await commands.claim_next()
    assert stale is not None and stale.kind == "approve"
    changed = await commands.reset_stale(datetime.now(UTC) + timedelta(minutes=1))
    assert changed == 1
    reclaimed = await commands.claim_next()
    assert reclaimed is not None and reclaimed.id == stale.id
    await engine.dispose()
