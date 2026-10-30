import pytest

from eva_agent.domain.run import RunStatus
from eva_agent.runtime.manager import RunManager, RunStateError
from eva_agent.storage.db import create_engine, create_session_factory
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import budget, dependencies, fake_config, paste_input


async def test_create_run_only_persists_a_queued_row(tmp_path) -> None:
    config = fake_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path)) as manager:
        run_id = await manager.create_run(
            paste_input(),
            budget(),
            idempotency_key="key-1",
            created_by="admin",
        )

    assert run_id.startswith("run_")
    assert len(run_id) == 36
    engine = create_engine(config.storage.url)
    row = await RunRepository(create_session_factory(engine)).get(run_id)
    assert row is not None
    assert row.status == RunStatus.QUEUED.value
    assert row.idempotency_key == "key-1"
    assert row.created_by == "admin"
    await engine.dispose()


async def test_start_existing_runs_the_graph_to_approval(tmp_path) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        run_id = await manager.create_run(paste_input(), budget())

        view = await manager.start_existing(run_id)

        assert view.status is RunStatus.WAITING_APPROVAL
        with pytest.raises(RunStateError, match="already has a checkpoint"):
            await manager.start_existing(run_id)


async def test_resume_without_a_checkpoint_is_rejected(tmp_path) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        run_id = await manager.create_run(paste_input(), budget())

        with pytest.raises(RunStateError, match="no checkpoint"):
            await manager.resume(run_id)


async def test_cancel_a_run_waiting_for_approval(tmp_path) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        run_id = await manager.create_run(paste_input(), budget())
        await manager.start_existing(run_id)

        view = await manager.cancel(run_id)

        assert view.status is RunStatus.CANCELLED
        assert (await manager.status(run_id)).status is RunStatus.CANCELLED


async def test_start_still_creates_and_runs_in_one_call(tmp_path) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        run_id = await manager.start(paste_input(), budget())

        assert (await manager.status(run_id)).status is RunStatus.WAITING_APPROVAL
