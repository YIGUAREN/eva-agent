from pathlib import Path

import pytest

from eva_agent.domain.grader import CheckSpec
from eva_agent.domain.probe import ProbeSummary
from eva_agent.domain.run import RunStatus
from eva_agent.domain.target import TargetSpec
from eva_agent.runtime.manager import RunManager
from tests.support.runtime import fake_config
from tests.support.target_runtime import (
    target_dependencies,
    target_plan,
    target_state,
)


async def test_target_run_round_trips_through_sqlite_checkpoints(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[str] = []
    state = target_state()
    edited = target_plan().model_dump(mode="json")
    edited["dimensions"][0]["sample_size"] = 1

    async with RunManager.open(
        fake_config(tmp_path),
        dependencies=target_dependencies(tmp_path, calls, events),
    ) as manager:
        await manager.start(state["input"], state["budget"], run_id="run-1")
        view = await manager.approve("run-1", approved=True, plan=edited)
        report = await manager.report("run-1")

    assert view.status is RunStatus.COMPLETED
    assert calls == ["a", "b"]
    assert report["probe"]["total"] == 2
    assert {item["dimension"] for item in report["dimensions"]} == {"latency", "cost"}


async def test_checkpoint_rehydrates_phase2_domain_objects(tmp_path: Path) -> None:
    """Blocked msgpack types degrade silently to dicts, so assert the real types come back."""
    calls: list[str] = []
    events: list[str] = []
    state = target_state()

    async with RunManager.open(
        fake_config(tmp_path),
        dependencies=target_dependencies(tmp_path, calls, events),
    ) as manager:
        await manager.start(state["input"], state["budget"], run_id="run-1")
        await manager.approve("run-1", approved=True)
        snapshot = await manager._require_graph().aget_state({"configurable": {"thread_id": "run-1"}})

    probe = snapshot.values["probe"]
    assert isinstance(probe, ProbeSummary)
    assert probe.total == 2
    evidence = snapshot.values["evidence"]
    assert isinstance(evidence.target, TargetSpec)
    assert evidence.target.name == "demo"
    plan = snapshot.values["plan"]
    assert isinstance(plan.dimensions[0].graders[0].checks[0], CheckSpec)


async def test_unknown_target_does_not_create_a_run(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[str] = []
    state = target_state()
    unknown = state["input"].model_copy(update={"target_ref": "missing"})

    async with RunManager.open(
        fake_config(tmp_path),
        dependencies=target_dependencies(tmp_path, calls, events),
    ) as manager:
        with pytest.raises(ValueError, match="unknown target 'missing'"):
            await manager.start(unknown, state["budget"], run_id="run-1")
        with pytest.raises(KeyError):
            await manager.status("run-1")
