import pytest

from eva_agent.runtime.manager import RunManager
from tests.support.runtime import budget, paste_input


def test_initial_state_defaults_to_no_run_authorization() -> None:
    state = RunManager._initial_state("run-1", paste_input(), budget())
    assert state["authorization_ref"] is None
    assert state["requester_is_admin"] is None


@pytest.mark.asyncio
async def test_start_existing_forwards_row_authorization(tmp_path, monkeypatch) -> None:
    from tests.support.runtime import dependencies, fake_config

    manager = RunManager(fake_config(tmp_path), dependencies=dependencies(tmp_path))
    await manager.setup()
    captured: dict = {}

    async def record(self, run_id, run_input, run_budget, **kwargs):
        captured["run_id"] = run_id
        captured.update(kwargs)

    try:
        run_id = await manager.create_run(
            paste_input(),
            budget(),
            authorization_ref="approval-42",
            created_by="admin",
        )
        monkeypatch.setattr(RunManager, "_invoke_initial", record)
        await manager.start_existing(run_id)
    finally:
        await manager.close()

    assert captured["run_id"] == run_id
    assert captured["authorization_ref"] == "approval-42"
    assert captured["requester_is_admin"] is True


@pytest.mark.asyncio
async def test_start_existing_leaves_non_admin_runs_unauthorized(tmp_path, monkeypatch) -> None:
    from tests.support.runtime import dependencies, fake_config

    manager = RunManager(fake_config(tmp_path), dependencies=dependencies(tmp_path))
    await manager.setup()
    captured: dict = {}

    async def record(self, run_id, run_input, run_budget, **kwargs):
        captured.update(kwargs)

    try:
        run_id = await manager.create_run(paste_input(), budget())
        monkeypatch.setattr(RunManager, "_invoke_initial", record)
        await manager.start_existing(run_id)
    finally:
        await manager.close()

    assert captured["authorization_ref"] is None
    assert captured["requester_is_admin"] is None
