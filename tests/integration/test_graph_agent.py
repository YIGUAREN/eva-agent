from dataclasses import replace
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.run import InputKind, RunInput, RunStatus
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.build import build_graph
from eva_agent.runtime.artifacts import run_paths
from tests.support.runtime import FakePlanner, dependencies
from tests.support.target_runtime import FakePolicy

CONFIG = {"configurable": {"thread_id": "run-agent"}}

AGENT = TargetSpec(
    name="demo-agent",
    kind=TargetKind.AGENT_HTTP,
    provider="dify",
    model="agent",
    url="http://127.0.0.1:8200/chat",
    method="POST",
    request_template={"message": "{{prompt}}"},
    response_path="$.answer",
)


def agent_plan() -> EvalPlan:
    return EvalPlan(
        objective="评估 agent",
        evidence_summary="agent 目标",
        dimensions=[
            DimensionPlan(dimension=Dimension.SECURITY, enabled=True, sample_size=1, graders=[]),
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.QUALITY, reason="agent 目标没有本地响应"),
            SkippedDimension(dimension=Dimension.LATENCY, reason="agent 扫描不产生本地 probe"),
            SkippedDimension(dimension=Dimension.COST, reason="agent 扫描不产生本地 probe"),
        ],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1),
    )


def agent_input() -> RunInput:
    return RunInput(kind=InputKind.TARGET, target_ref="demo-agent", target_prompts=["hi"])


async def test_agent_target_skips_probe(tmp_path: Path) -> None:
    deps = replace(
        dependencies(tmp_path, planner=FakePlanner(plan=agent_plan())),
        targets={"demo-agent": AGENT},
        policy=FakePolicy(),
    )
    app = build_graph(deps, MemorySaver())
    result = await app.ainvoke(
        {
            "run_id": "run-agent",
            "input": agent_input(),
            "budget": agent_plan().budget,
            "evidence": None,
            "plan": None,
            "probe": None,
            "status": "queued",
            "errors": [],
            "approval": None,
            "expert_results": [],
            "merged": None,
            "report": None,
            "authorization_ref": "demo-agent",
            "requester_is_admin": True,
        },
        CONFIG,
    )
    assert result["status"] is RunStatus.WAITING_APPROVAL
    assert result["evidence"].kind is InputKind.TARGET
    assert result["evidence"].target is not None
    assert result["evidence"].target.kind is TargetKind.AGENT_HTTP

    await app.ainvoke(Command(resume={"approved": True}), CONFIG)
    snapshot = await app.aget_state(CONFIG)

    # agent_scan runs entirely inside AIG; the graph must not probe locally.
    assert snapshot.values["probe"] is None
    assert not run_paths(tmp_path / "data", "run-agent").target_probe.exists()
    assert [item.dimension for item in snapshot.values["expert_results"]] == [Dimension.SECURITY]
