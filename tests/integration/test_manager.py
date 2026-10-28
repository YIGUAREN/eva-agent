from pathlib import Path

import pytest

from eva_agent.domain.plan import Dimension
from eva_agent.domain.run import RunStatus
from eva_agent.graph.plan_validation import PlanValidationError
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.manager import RunManager, RunStateError
from tests.support.runtime import (
    FakePlanner,
    FakeRunner,
    budget,
    dependencies,
    fake_config,
    paste_input,
    sample_plan,
)


async def test_start_waits_for_approval(tmp_path: Path) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        run_id = await manager.start(paste_input(), budget(), run_id="run-1")
        view = await manager.status(run_id)

    assert view.status is RunStatus.WAITING_APPROVAL
    assert view.plan is not None
    assert view.report_json_path is None
    assert run_paths(tmp_path / "data", "run-1").plan_draft.is_file()


async def test_approval_completes_and_writes_report(tmp_path: Path) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        await manager.start(paste_input(), budget(), run_id="run-1")
        view = await manager.approve("run-1", approved=True)
        report = await manager.report("run-1")

    assert view.status is RunStatus.COMPLETED
    assert view.report_json_path is not None
    assert report["status"] == "completed"
    assert report["dimensions"][0]["dimension"] == "quality"
    assert run_paths(tmp_path / "data", "run-1").plan_approved.is_file()


async def test_rejection_cancels_without_grading(tmp_path: Path) -> None:
    runner = FakeRunner()
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path, runner=runner)) as manager:
        await manager.start(paste_input(), budget(), run_id="run-1")
        view = await manager.approve("run-1", approved=False)

    assert view.status is RunStatus.CANCELLED
    assert runner.calls == 0
    assert view.report_json_path is None


async def test_invalid_plan_edit_is_rejected_without_consuming_interrupt(tmp_path: Path) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        await manager.start(paste_input(), budget(), run_id="run-1")
        edited = sample_plan().model_dump(mode="json")
        edited["objective"] = "另一个目标"

        with pytest.raises(PlanValidationError):
            await manager.approve("run-1", approved=True, plan=edited)
        waiting = await manager.status("run-1")
        assert waiting.status is RunStatus.WAITING_APPROVAL

        valid = sample_plan().model_dump(mode="json")
        valid["dimensions"][0]["sample_size"] = 2
        valid["budget"]["max_cost_usd"] = 2.0
        view = await manager.approve("run-1", approved=True, plan=valid)

    assert view.status is RunStatus.COMPLETED
    assert view.plan is not None
    assert view.plan.budget.max_cost_usd == 2.0


async def test_unknown_run_raises(tmp_path: Path) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        with pytest.raises(KeyError):
            await manager.status("missing")
        with pytest.raises(KeyError):
            await manager.report("missing")


async def test_approve_requires_waiting_state(tmp_path: Path) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        await manager.start(paste_input(), budget(), run_id="run-1")
        await manager.approve("run-1", approved=True)

        with pytest.raises(RunStateError):
            await manager.approve("run-1", approved=True)


async def test_restart_recovers_completed_run(tmp_path: Path) -> None:
    config = fake_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path)) as manager:
        await manager.start(paste_input(), budget(), run_id="run-1")
        await manager.approve("run-1", approved=True)

    assert (tmp_path / "checkpoints.sqlite").is_file()

    async with RunManager.open(config, dependencies=dependencies(tmp_path)) as reopened:
        view = await reopened.status("run-1")
        report = await reopened.report("run-1")

    assert view.status is RunStatus.COMPLETED
    assert view.plan is not None
    assert view.plan.dimensions[0].dimension is Dimension.QUALITY
    assert report["status"] == "completed"


async def test_planner_failure_marks_run_failed(tmp_path: Path) -> None:
    planner = FakePlanner(error=RuntimeError("boom"))
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path, planner=planner)) as manager:
        await manager.start(paste_input(), budget(), run_id="run-1")
        view = await manager.status("run-1")

    assert view.status is RunStatus.FAILED
    assert view.errors == ["制定评估计划失败: boom"]


async def test_event_sink_failure_is_logged_not_raised(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        await manager.start(paste_input(), budget(), run_id="run-1")
        repo = manager._require_repo()

        async def failing_append(*args: object, **kwargs: object) -> None:
            raise RuntimeError("db down")

        repo.append_event = failing_append  # type: ignore[method-assign]
        await manager._event_sink("run-1", "plan.ready", {})

    assert "db down" in caplog.text
