"""Shared target-run fakes for graph and manager integration tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import CheckSpec, GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import InputKind, RunInput
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.pricing import ModelPrice, PricingTable
from eva_agent.security.target_policy import ValidatedTarget
from eva_agent.tools.declarative import build_run_declarative_checks_tool
from eva_agent.tools.pricing import build_get_pricing_tool
from eva_agent.tools.registry import ToolRegistry, ToolSpec
from eva_agent.tools.target import CallTargetInput, build_read_target_probe_tool
from tests.support.runtime import FakePlanner, initial_state


def demo_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )


def declarative(name: str, field: str, threshold: float) -> GraderSpec:
    return GraderSpec(
        level="declarative",
        name=name,
        input_map={field: f"$.{field}", "response": "$.response"},
        checks=[CheckSpec(type="numeric_threshold", field=field, op="<=", value=threshold)],
    )


def target_plan() -> EvalPlan:
    return EvalPlan(
        objective="评估 demo target",
        evidence_summary="2 条 prompt",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.LATENCY,
                enabled=True,
                sample_size=2,
                graders=[declarative("latency_budget", "latency_ms", 1000)],
            ),
            DimensionPlan(
                dimension=Dimension.COST,
                enabled=True,
                sample_size=2,
                graders=[declarative("cost_budget", "cost_usd", 0.01)],
            ),
        ],
        skipped=[
            SkippedDimension(
                dimension=Dimension.QUALITY,
                reason="阶段 2 不支持 target 输入上的 quality",
            ),
            SkippedDimension(dimension=Dimension.SECURITY, reason="阶段 3 实现"),
        ],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
    )


def target_budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10)


class FakePolicy:
    async def validate(self, target: TargetSpec) -> ValidatedTarget:
        return ValidatedTarget(
            url="http://127.0.0.1:8100/v1",
            scheme="http",
            host="127.0.0.1",
            port=8100,
            addresses=("127.0.0.1",),
            demo=True,
        )


def build_tools(calls: list[str], *, pricing_error: str | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    async def target_handler(payload: CallTargetInput) -> ProbeRecord:
        calls.append(payload.prompt)
        return ProbeRecord(
            index=payload.index,
            prompt=payload.prompt,
            started_at=datetime.now(UTC),
            status_code=200,
            latency_ms=10,
            response_text="ok",
            input_tokens=1000,
            output_tokens=500,
            usage_source="target",
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
            handler=target_handler,
        )
    )
    registry.register(build_read_target_probe_tool())
    registry.register(build_run_declarative_checks_tool(max_rows=100))
    pricing = build_get_pricing_tool(
        PricingTable(
            [
                ModelPrice(
                    provider="demo",
                    model="demo-echo",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                )
            ]
        )
    )
    if pricing_error is not None:

        async def failing_pricing(_: Any) -> Any:
            raise RuntimeError(pricing_error)

        # Only the cost expert resolves pricing, so this fails exactly one dimension.
        pricing = replace(pricing, handler=failing_pricing)
    registry.register(pricing)
    return registry


def target_dependencies(
    tmp_path: Path,
    calls: list[str],
    events: list[str],
    *,
    pricing_error: str | None = None,
) -> GraphDependencies:
    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append(event_type)

    return GraphDependencies(
        planner=FakePlanner(plan=target_plan()),
        tools=build_tools(calls, pricing_error=pricing_error),
        catalog_ids=frozenset(),
        data_dir=str(tmp_path / "data"),
        max_input_bytes=1_000_000,
        max_dataset_rows=100,
        default_sample_size=5,
        event_sink=sink,
        targets={"demo": demo_target()},
        policy=cast(Any, FakePolicy()),
        env={},
        target_timeout_s=5.0,
        max_response_bytes=1024,
    )


def target_state() -> dict:
    return {
        **initial_state(
            RunInput(
                kind=InputKind.TARGET,
                target_ref="demo",
                target_prompts=["a", "b"],
            )
        ),
        "budget": target_budget(),
    }
