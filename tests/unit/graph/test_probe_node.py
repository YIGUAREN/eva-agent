from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import InputKind, RunInput
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.nodes.probe import probe_node
from eva_agent.graph.state import EvalState
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.probe import ProbeStore
from eva_agent.tools.registry import ToolRegistry, ToolSpec
from eva_agent.tools.target import CallTargetInput
from tests.support.runtime import FakePlanner, initial_state


def target_input(prompts: list[str]) -> RunInput:
    return RunInput(kind=InputKind.TARGET, target_ref="demo", target_prompts=prompts)


def demo_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )


def target_plan(*, latency_size: int = 3, cost_size: int = 2) -> EvalPlan:
    return EvalPlan(
        objective="评估目标",
        evidence_summary="目标证据",
        dimensions=[
            DimensionPlan(dimension=Dimension.LATENCY, enabled=True, sample_size=latency_size),
            DimensionPlan(dimension=Dimension.COST, enabled=True, sample_size=cost_size),
        ],
        skipped=[],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
    )


def build_tools(
    calls: list[str],
    *,
    fail_index: int | None = None,
    exception_index: int | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()

    async def handler(payload: CallTargetInput) -> ProbeRecord:
        calls.append(payload.prompt)
        if payload.index == exception_index:
            raise RuntimeError("boom")
        failed = payload.index == fail_index
        return ProbeRecord(
            index=payload.index,
            prompt=payload.prompt,
            started_at=datetime.now(UTC),
            status_code=500 if failed else 200,
            latency_ms=10,
            response_text="" if failed else "ok",
            error="http 500" if failed else None,
        )

    registry.register(
        ToolSpec(
            name="call_target",
            description="fake target call",
            input_model=CallTargetInput,
            output_model=ProbeRecord,
            permission="admin",
            timeout_s=10.0,
            side_effects=True,
            idempotent=False,
            handler=handler,
        )
    )
    return registry


def build_deps(tmp_path: Path, tools: ToolRegistry, events: list[tuple[str, dict]]) -> GraphDependencies:
    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    return GraphDependencies(
        planner=FakePlanner(),
        tools=tools,
        catalog_ids=frozenset(),
        data_dir=str(tmp_path / "data"),
        max_input_bytes=1_000_000,
        max_dataset_rows=100,
        default_sample_size=5,
        event_sink=sink,
        target_timeout_s=5.0,
        max_response_bytes=1024,
    )


def build_state(
    tmp_path: Path,
    prompts: list[str],
    plan: EvalPlan,
    *,
    max_target_requests: int = 10,
) -> EvalState:
    evidence = EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(tmp_path / "data" / "runs" / "run-1" / "evidence"),
        target=demo_target(),
        target_prompts=prompts,
        manifest_sha256="a" * 64,
    )
    return cast(
        EvalState,
        {
            **initial_state(target_input(prompts)),
            "budget": Budget(
                max_cost_usd=1.0,
                max_wall_clock_s=60.0,
                max_target_requests=max_target_requests,
            ),
            "plan": plan,
            "evidence": evidence,
        },
    )


async def test_probe_uses_max_sample_size_and_cycles_prompts(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[tuple[str, dict]] = []
    tools = build_tools(calls)
    state = build_state(tmp_path, ["a", "b"], target_plan(latency_size=3, cost_size=2))

    result = await probe_node(state, build_deps(tmp_path, tools, events))

    assert calls == ["a", "b", "a"]
    assert result["probe"].total == 3
    assert result["probe"].succeeded == 3
    assert [item.index for item in ProbeStore(run_paths(tmp_path / "data", "run-1").target_probe).read()] == [0, 1, 2]
    assert events[0] == ("probe.started", {"target": "demo", "requests": 3})
    assert [event for event, _ in events].count("probe.record") == 3
    assert events[-1] == ("probe.finished", {"total": 3, "succeeded": 3, "failed": 0})


async def test_one_failed_request_does_not_stop_the_probe(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[tuple[str, dict]] = []
    tools = build_tools(calls, fail_index=1)
    state = build_state(tmp_path, ["a"], target_plan(latency_size=3, cost_size=2))

    result = await probe_node(state, build_deps(tmp_path, tools, events))

    assert calls == ["a", "a", "a"]
    assert result["probe"].total == 3
    assert result["probe"].succeeded == 2
    assert result["probe"].failed == 1
    assert events[-1] == ("probe.finished", {"total": 3, "succeeded": 2, "failed": 1})
    records = ProbeStore(run_paths(tmp_path / "data", "run-1").target_probe).read()
    assert records[1].status_code == 500
    assert records[1].error == "http 500"


async def test_tool_exception_is_recorded_and_the_probe_continues(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[tuple[str, dict]] = []
    tools = build_tools(calls, exception_index=1)
    state = build_state(tmp_path, ["a"], target_plan(latency_size=3, cost_size=2))

    result = await probe_node(state, build_deps(tmp_path, tools, events))

    assert calls == ["a", "a", "a"]
    assert result["probe"].failed == 1
    records = ProbeStore(run_paths(tmp_path / "data", "run-1").target_probe).read()
    assert records[1].error == "RuntimeError: boom"
    assert records[1].status_code is None


async def test_probe_rejects_more_requests_than_the_budget(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[tuple[str, dict]] = []
    tools = build_tools(calls)
    state = build_state(
        tmp_path,
        ["a"],
        target_plan(latency_size=3, cost_size=2),
        max_target_requests=2,
    )

    with pytest.raises(ValueError, match="max_target_requests"):
        await probe_node(state, build_deps(tmp_path, tools, events))

    assert calls == []
    assert events == []


async def test_probe_requires_a_sample_size_for_every_target_dimension(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[tuple[str, dict]] = []
    plan = EvalPlan(
        objective="评估目标",
        evidence_summary="目标证据",
        dimensions=[
            DimensionPlan(dimension=Dimension.LATENCY, enabled=True, sample_size=2),
            DimensionPlan(dimension=Dimension.COST, enabled=True, sample_size=None),
        ],
        skipped=[],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
    )
    state = build_state(tmp_path, ["a"], plan)

    with pytest.raises(ValueError, match="sample_size"):
        await probe_node(state, build_deps(tmp_path, build_tools(calls), events))


async def test_probe_overwrites_the_previous_probe_file(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[tuple[str, dict]] = []
    paths = run_paths(tmp_path / "data", "run-1")
    store = ProbeStore(paths.target_probe)
    store.reset()
    store.append(
        ProbeRecord(
            index=99,
            prompt="old",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status_code=200,
            latency_ms=1,
        )
    )
    state = build_state(tmp_path, ["a"], target_plan(latency_size=2, cost_size=1))

    await probe_node(state, build_deps(tmp_path, build_tools(calls), events))

    assert [item.index for item in store.read()] == [0, 1]
