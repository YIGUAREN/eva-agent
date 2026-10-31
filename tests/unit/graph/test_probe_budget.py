"""Probe request reservation, exhaustion, and unlimited behavior."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import InputKind, RunInput, RunStatus
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.nodes.probe import probe_node
from eva_agent.graph.state import EvalState
from eva_agent.tools.registry import ToolRegistry, ToolSpec
from eva_agent.tools.target import CallTargetInput
from tests.support.runtime import FakePlanner, initial_state


class NoopStore:
    async def load(self, run_id: str) -> None:
        return None

    async def save(self, run_id: str, usage: Any) -> None:
        return None


def model() -> ModelConfig:
    return ModelConfig(
        provider="litellm",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
        api_key_env="DEMO_KEY",
    )


def bound_models() -> BudgetModels:
    return BudgetModels(agent=model(), judge=model(), attack=model())


def demo_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )


def target_plan() -> EvalPlan:
    return EvalPlan(
        objective="评估目标",
        evidence_summary="目标证据",
        dimensions=[
            DimensionPlan(dimension=Dimension.LATENCY, enabled=True, sample_size=3),
            DimensionPlan(dimension=Dimension.COST, enabled=True, sample_size=2),
        ],
        skipped=[],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
    )


def build_tools(calls: list[str]) -> ToolRegistry:
    registry = ToolRegistry()

    async def handler(payload: CallTargetInput) -> ProbeRecord:
        calls.append(payload.prompt)
        return ProbeRecord(
            index=payload.index,
            prompt=payload.prompt,
            started_at=datetime.now(UTC),
            status_code=200,
            latency_ms=5,
            response_text="ok",
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


def build_state(tmp_path: Path) -> EvalState:
    evidence = EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(tmp_path / "data" / "runs" / "run-1" / "evidence"),
        target=demo_target(),
        target_prompts=["a"],
        manifest_sha256="a" * 64,
    )
    return cast(
        EvalState,
        {
            **initial_state(RunInput(kind=InputKind.TARGET, target_ref="demo", target_prompts=["a"])),
            "budget": Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
            "plan": target_plan(),
            "evidence": evidence,
        },
    )


def budget_events(events: list[tuple[str, dict]]) -> list[dict]:
    return [payload for event_type, payload in events if event_type == "budget.exhausted"]


async def test_probe_reserves_the_request_count(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[tuple[str, dict]] = []
    tools = build_tools(calls)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
        store=NoopStore(),
    )
    with bind_budget(ledger, CostEstimator(None), bound_models()):
        result = await probe_node(build_state(tmp_path), build_deps(tmp_path, tools, events))
        snapshot = ledger.snapshot()

    assert result["probe"].total == 3
    assert snapshot.target_requests == 3
    assert snapshot.aig_requests == 0
    assert snapshot.spent_usd == Decimal("0")
    assert snapshot.estimated is False
    assert budget_events(events) == []


async def test_probe_exhaustion_fails_before_calling_the_target(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[tuple[str, dict]] = []
    tools = build_tools(calls)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=2),
        store=NoopStore(),
    )
    with bind_budget(ledger, CostEstimator(None), bound_models()):
        result = await probe_node(build_state(tmp_path), build_deps(tmp_path, tools, events))
        snapshot = ledger.snapshot()

    assert result == {"status": RunStatus.FAILED, "errors": ["目标请求预算不足"]}
    assert calls == []
    assert snapshot.target_requests == 0
    assert budget_events(events) == [
        {"spent_usd": 0.0, "remaining_usd": 1.0, "dimension": "probe"}
    ]


async def test_unlimited_ledger_leaves_the_probe_counters_untouched(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[tuple[str, dict]] = []
    tools = build_tools(calls)
    ledger = BudgetLedger.unlimited(Budget(max_cost_usd=0.01, max_wall_clock_s=1.0, max_target_requests=1))
    with bind_budget(ledger, CostEstimator(None), bound_models()):
        result = await probe_node(build_state(tmp_path), build_deps(tmp_path, tools, events))
        snapshot = ledger.snapshot()

    assert result["probe"].total == 3
    assert snapshot.target_requests == 0
    assert snapshot.aig_requests == 0
    assert snapshot.spent_usd == Decimal("0")
    assert snapshot.estimated is False
    assert budget_events(events) == []
