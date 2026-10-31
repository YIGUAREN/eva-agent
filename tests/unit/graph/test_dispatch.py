from typing import cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.probe import ProbeSummary
from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.build import build_graph, route_experts
from eva_agent.graph.nodes.dispatch import dispatch_node
from eva_agent.graph.state import EvalState
from eva_agent.runtime.artifacts import run_paths
from tests.support.runtime import (
    FakePlanner,
    dependencies,
    initial_state,
    paste_input,
    sample_plan,
)

CONFIG = {"configurable": {"thread_id": "run-1"}}


def all_skipped_plan() -> EvalPlan:
    return sample_plan().model_copy(
        update={
            "dimensions": [],
            "skipped": [
                SkippedDimension(dimension=dimension, reason="证据没有可用响应") for dimension in Dimension
            ],
        }
    )


def target_plan(*dimensions: DimensionPlan) -> EvalPlan:
    return sample_plan().model_copy(update={"dimensions": list(dimensions)})


def target_plan_two() -> EvalPlan:
    return target_plan(
        DimensionPlan(dimension=Dimension.LATENCY, enabled=True, sample_size=2),
        DimensionPlan(dimension=Dimension.COST, enabled=True, sample_size=1),
    )


async def test_dispatch_accepts_target_dimensions(tmp_path) -> None:
    events: list[tuple[str, dict]] = []

    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    deps = dependencies(tmp_path, event_sink=sink)
    plan = target_plan(
        DimensionPlan(dimension=Dimension.LATENCY, enabled=True, sample_size=2),
        DimensionPlan(dimension=Dimension.COST, enabled=True, sample_size=1),
    )
    state = cast(EvalState, {**initial_state(paste_input()), "plan": plan})

    result = await dispatch_node(state, deps)

    assert result["status"] is RunStatus.RUNNING
    assert events == [("dispatch.ready", {"dimensions": ["latency", "cost"]})]


async def test_dispatch_accepts_security(tmp_path) -> None:
    events: list[tuple[str, dict]] = []

    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    deps = dependencies(tmp_path, event_sink=sink)
    plan = target_plan(DimensionPlan(dimension=Dimension.SECURITY, enabled=True, sample_size=1))
    state = cast(EvalState, {**initial_state(paste_input()), "plan": plan})

    result = await dispatch_node(state, deps)

    assert result["status"] is RunStatus.RUNNING
    assert events == [("dispatch.ready", {"dimensions": ["security"]})]


async def test_dispatch_accepts_an_all_skipped_plan(tmp_path) -> None:
    events: list[tuple[str, dict]] = []

    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    deps = dependencies(tmp_path, event_sink=sink)
    state = cast(EvalState, {**initial_state(paste_input()), "plan": all_skipped_plan()})

    result = await dispatch_node(state, deps)

    assert result["status"] is RunStatus.RUNNING
    assert events == [("dispatch.ready", {"dimensions": []})]


def test_route_experts_merges_an_all_skipped_plan() -> None:
    state = cast(
        EvalState,
        {
            **initial_state(paste_input()),
            "plan": all_skipped_plan(),
            "status": RunStatus.RUNNING,
        },
    )

    assert route_experts(state) == "merge"


async def test_all_skipped_run_completes_with_skipped_annotations(tmp_path) -> None:
    events: list[str] = []

    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append(event_type)

    app = build_graph(
        dependencies(tmp_path, planner=FakePlanner(plan=all_skipped_plan()), event_sink=sink),
        MemorySaver(),
    )
    await app.ainvoke(initial_state(paste_input()), CONFIG)

    await app.ainvoke(Command(resume={"approved": True}), CONFIG)
    snapshot = await app.aget_state(CONFIG)

    assert snapshot.values["status"] is RunStatus.COMPLETED
    assert snapshot.values["merged"].status is RunStatus.COMPLETED
    assert snapshot.values["merged"].metrics["overall_score"] is None
    assert snapshot.values["merged"].metrics["dimensions"] == {}
    assert {item.dimension for item in snapshot.values["merged"].skipped} == set(Dimension)
    assert snapshot.values["expert_results"] == []
    assert run_paths(tmp_path / "data", "run-1").report_json.is_file()
    assert {"dispatch.ready", "run.merged", "report.ready"} <= set(events)


def test_route_experts_sends_one_payload_per_dimension() -> None:
    evidence = EvidenceBundle(
        kind=InputKind.TARGET,
        root="/tmp/evidence",
        target=TargetSpec(
            name="demo",
            kind=TargetKind.OPENAI_COMPATIBLE,
            provider="demo",
            model="demo-echo",
            base_url="http://127.0.0.1:8100/v1",
        ),
        target_prompts=["a"],
        manifest_sha256="a" * 64,
    )
    probe = ProbeSummary(path="target_probe.jsonl", total=2, succeeded=2, failed=0)
    state = cast(
        EvalState,
        {
            **initial_state(paste_input()),
            "budget": Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
            "plan": target_plan_two(),
            "evidence": evidence,
            "probe": probe,
            "status": RunStatus.RUNNING,
        },
    )

    sends = route_experts(state)

    assert isinstance(sends, list)
    assert [send.node for send in sends] == ["latency", "cost"]
    assert sends[0].arg["dimension_plan"].dimension is Dimension.LATENCY
    assert sends[0].arg["probe"] == probe
    assert sends[0].arg["run_id"] == "run-1"


def test_route_experts_returns_end_when_dispatch_failed() -> None:
    state = cast(
        EvalState,
        {
            **initial_state(paste_input()),
            "plan": target_plan(DimensionPlan(dimension=Dimension.SECURITY, enabled=True)),
            "status": RunStatus.FAILED,
        },
    )

    assert route_experts(state) == END


async def test_parallel_failing_branches_merge_status_without_conflict() -> None:
    """Two experts reporting FAILED in one superstep must not raise InvalidUpdateError."""

    async def start(state: EvalState) -> dict:
        return {"status": RunStatus.RUNNING}

    async def fail(state: EvalState) -> dict:
        return {"status": RunStatus.FAILED, "errors": ["专家执行失败"]}

    def fan_out(state: EvalState) -> list[Send]:
        return [Send("left", {}), Send("right", {})]

    graph = StateGraph(EvalState)
    graph.add_node("start", start)
    graph.add_node("left", fail)
    graph.add_node("right", fail)
    graph.add_edge(START, "start")
    graph.add_conditional_edges("start", fan_out, {"left": "left", "right": "right"})
    graph.add_edge("left", END)
    graph.add_edge("right", END)

    result = await graph.compile().ainvoke(initial_state(paste_input()))

    assert result["status"] is RunStatus.FAILED
    assert result["errors"] == ["专家执行失败", "专家执行失败"]
