from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.graph.build import build_graph
from eva_agent.runtime.artifacts import run_paths
from tests.support.runtime import (
    FakePlanner,
    FakeRunner,
    dataset_input,
    dependencies,
    initial_state,
    paste_input,
    sample_plan,
)

CONFIG = {"configurable": {"thread_id": "run-1"}}


async def test_paste_run_stops_for_approval(tmp_path: Path) -> None:
    planner = FakePlanner()
    app = build_graph(dependencies(tmp_path, planner=planner), MemorySaver())

    result = await app.ainvoke(initial_state(paste_input()), CONFIG)

    assert result["status"] is RunStatus.WAITING_APPROVAL
    assert result["plan"] is not None
    assert "__interrupt__" in result
    snapshot = await app.aget_state(CONFIG)
    assert snapshot.next == ("approve",)
    paths = run_paths(tmp_path / "data", "run-1")
    assert paths.plan_draft.is_file()
    assert planner.calls == 1


async def test_approval_completes_run_and_writes_report(tmp_path: Path) -> None:
    events: list[str] = []

    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append(event_type)

    app = build_graph(dependencies(tmp_path, event_sink=sink), MemorySaver())
    await app.ainvoke(initial_state(paste_input()), CONFIG)

    await app.ainvoke(Command(resume={"approved": True}), CONFIG)
    snapshot = await app.aget_state(CONFIG)

    assert snapshot.values["status"] is RunStatus.COMPLETED
    assert snapshot.values["merged"].status is RunStatus.COMPLETED
    assert len(snapshot.values["expert_results"]) == 1
    assert snapshot.values["expert_results"][0].score == 0.8
    paths = run_paths(tmp_path / "data", "run-1")
    assert paths.plan_approved.is_file()
    assert paths.report_json.is_file()
    assert {
        "evidence.ready",
        "plan.ready",
        "plan.approved",
        "dispatch.ready",
        "quality.completed",
        "run.merged",
        "report.ready",
    } <= set(events)


async def test_rejection_cancels_without_grading(tmp_path: Path) -> None:
    runner = FakeRunner()
    app = build_graph(dependencies(tmp_path, runner=runner), MemorySaver())
    await app.ainvoke(initial_state(paste_input()), CONFIG)

    await app.ainvoke(Command(resume={"approved": False}), CONFIG)
    snapshot = await app.aget_state(CONFIG)

    assert snapshot.values["status"] is RunStatus.CANCELLED
    assert snapshot.values["expert_results"] == []
    assert runner.calls == 0
    assert not run_paths(tmp_path / "data", "run-1").report_json.exists()


async def test_invalid_plan_edit_fails_run(tmp_path: Path) -> None:
    app = build_graph(dependencies(tmp_path), MemorySaver())
    await app.ainvoke(initial_state(paste_input()), CONFIG)
    edited = sample_plan().model_dump(mode="json")
    edited["objective"] = "另一个目标"

    await app.ainvoke(Command(resume={"approved": True, "plan": edited}), CONFIG)
    snapshot = await app.aget_state(CONFIG)

    assert snapshot.values["status"] is RunStatus.FAILED
    assert any("计划修改无效" in error for error in snapshot.values["errors"])
    assert snapshot.values["plan"] is not None


async def test_upload_is_rejected_without_calling_planner(tmp_path: Path) -> None:
    planner = FakePlanner()
    app = build_graph(dependencies(tmp_path, planner=planner))
    state = initial_state(paste_input())
    state["input"] = state["input"].model_copy(update={"kind": InputKind.UPLOAD, "paste": None, "upload_id": "u1"})

    result = await app.ainvoke(state)

    assert result["status"] is RunStatus.FAILED
    assert "not supported in phase 1" in result["errors"][0]
    assert planner.calls == 0


async def test_planner_exception_fails_run(tmp_path: Path) -> None:
    app = build_graph(dependencies(tmp_path, planner=FakePlanner(error=RuntimeError("boom"))))

    result = await app.ainvoke(initial_state(paste_input()))

    assert result["status"] is RunStatus.FAILED
    assert result["errors"] == ["制定评估计划失败: boom"]


async def test_dataset_run_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("query,response,reference\nq1,a1,r1\nq2,a2,r2\n", encoding="utf-8")
    app = build_graph(dependencies(tmp_path), MemorySaver())
    state = initial_state(dataset_input(source))

    result = await app.ainvoke(state, CONFIG)
    assert result["status"] is RunStatus.WAITING_APPROVAL
    evidence = result["evidence"]
    assert evidence.kind is InputKind.DATASET
    assert evidence.root is not None
    assert evidence.dataset is not None
    assert evidence.dataset.path == "dataset.csv"
    assert Path(evidence.root, "dataset.csv").is_file()

    await app.ainvoke(Command(resume={"approved": True}), CONFIG)
    snapshot = await app.aget_state(CONFIG)

    assert snapshot.values["status"] is RunStatus.COMPLETED
    assert snapshot.values["expert_results"][0].usage.samples == 2


async def test_quality_failure_still_writes_report(tmp_path: Path) -> None:
    app = build_graph(
        dependencies(tmp_path, planner=FakePlanner(plan=sample_plan())),
        MemorySaver(),
    )
    await app.ainvoke(initial_state(paste_input(query="q", response=None)), CONFIG)

    await app.ainvoke(Command(resume={"approved": True}), CONFIG)
    snapshot = await app.aget_state(CONFIG)

    assert snapshot.values["status"] is RunStatus.FAILED
    assert snapshot.values["expert_results"][0].status == "failed"
    assert run_paths(tmp_path / "data", "run-1").report_json.is_file()
