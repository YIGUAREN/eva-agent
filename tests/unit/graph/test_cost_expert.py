from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.grader import CheckSpec, GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.experts.cost import CostExpert
from eva_agent.pricing import ModelPrice, PricingError, PricingTable
from eva_agent.runtime.probe import ProbeStore
from eva_agent.tools.declarative import build_run_declarative_checks_tool
from eva_agent.tools.pricing import build_get_pricing_tool
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.target import build_read_target_probe_tool


def record(
    index: int,
    *,
    input_tokens: int | None = 1000,
    output_tokens: int | None = 500,
    usage_source: Literal["target", "estimated", "missing"] = "target",
) -> ProbeRecord:
    return ProbeRecord(
        index=index,
        prompt="q",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        status_code=200,
        latency_ms=10,
        response_text="ok",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_source=usage_source,
    )


def demo_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )


def probe_evidence(tmp_path: Path, records: list[ProbeRecord]) -> EvidenceBundle:
    run_root = tmp_path / "runs" / "run-1"
    store = ProbeStore(run_root / "target_probe.jsonl")
    store.reset()
    for item in records:
        store.append(item)
    return EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(run_root / "evidence"),
        target=demo_target(),
        target_prompts=["q"],
        manifest_sha256="a" * 64,
    )


def price_table(*, include_demo: bool = True) -> PricingTable:
    entries = []
    if include_demo:
        entries.append(
            ModelPrice(
                provider="demo",
                model="demo-echo",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("4"),
            )
        )
    return PricingTable(entries)


def cost_plan(*, sample_size: int = 2, threshold: float = 0.01) -> DimensionPlan:
    return DimensionPlan(
        dimension=Dimension.COST,
        enabled=True,
        sample_size=sample_size,
        graders=[
            GraderSpec(
                level="declarative",
                name="cost_budget",
                input_map={"cost_usd": "$.cost_usd", "response": "$.response"},
                checks=[
                    CheckSpec(
                        type="numeric_threshold",
                        field="cost_usd",
                        op="<=",
                        value=threshold,
                    )
                ],
            )
        ],
    )


def build_tools(table: PricingTable | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_read_target_probe_tool())
    registry.register(build_get_pricing_tool(table or price_table()))
    registry.register(build_run_declarative_checks_tool(max_rows=100))
    return registry


async def run_expert(
    tmp_path: Path,
    records: list[ProbeRecord],
    plan: DimensionPlan,
    table: PricingTable | None = None,
):
    evidence = probe_evidence(tmp_path, records)
    return await CostExpert(build_tools(table)).run(
        run_id="run-1",
        plan=plan,
        evidence=evidence,
    )


async def test_costs_are_priced_and_aggregated(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0), record(1, input_tokens=2000, output_tokens=1000)],
        cost_plan(sample_size=2),
    )

    assert result.dimension is Dimension.COST
    assert result.status == "ok"
    assert result.score == 1.0
    assert result.confidence == 1.0
    assert result.metrics["count"] == 2
    assert result.metrics["priced_samples"] == 2
    assert result.metrics["estimated_samples"] == 0
    assert result.metrics["missing_usage_samples"] == 0
    assert result.metrics["input_tokens_total"] == 3000
    assert result.metrics["output_tokens_total"] == 1500
    assert result.metrics["cost_total_usd"] == 0.012
    assert result.metrics["cost_mean_usd"] == 0.006
    assert result.metrics["cost_per_1k_requests_usd"] == 6.0
    assert result.metrics["grader_scores"] == {"cost_budget": 1.0}
    assert result.usage.input_tokens == 3000
    assert result.usage.output_tokens == 1500
    assert result.evidence[0].locator == "target_probe.jsonl"


async def test_estimated_tokens_are_counted(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0, usage_source="estimated"), record(1)],
        cost_plan(sample_size=2),
    )

    assert result.status == "ok"
    assert result.metrics["estimated_samples"] == 1


async def test_missing_usage_makes_the_dimension_partial(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0), record(1, input_tokens=None, output_tokens=None, usage_source="missing")],
        cost_plan(sample_size=2),
    )

    assert result.status == "partial"
    assert result.score == 1.0
    assert result.confidence == 0.5
    assert result.metrics["priced_samples"] == 1
    assert result.metrics["missing_usage_samples"] == 1
    assert result.metrics["cost_total_usd"] == 0.004


async def test_no_priced_samples_fails(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [
            record(0, input_tokens=None, output_tokens=None, usage_source="missing"),
            record(1, input_tokens=None, output_tokens=None, usage_source="missing"),
        ],
        cost_plan(sample_size=2),
    )

    assert result.status == "failed"
    assert result.score is None
    assert result.confidence == 0.0
    assert result.error == "no samples with token usage"
    assert result.metrics["cost_mean_usd"] is None


async def test_declarative_cost_check_drives_the_score(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0), record(1, input_tokens=2000, output_tokens=1000)],
        cost_plan(sample_size=2, threshold=0.005),
    )

    assert result.score == 0.5
    assert result.metrics["grader_scores"] == {"cost_budget": 0.5}


async def test_unknown_pricing_fails_closed(tmp_path: Path) -> None:
    evidence = probe_evidence(tmp_path, [record(0)])

    with pytest.raises(PricingError, match="provider='demo'.*model='demo-echo'"):
        await CostExpert(build_tools(price_table(include_demo=False))).run(
            run_id="run-1",
            plan=cost_plan(sample_size=1),
            evidence=evidence,
        )


async def test_sample_size_limits_the_records(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0), record(1), record(2)],
        cost_plan(sample_size=1),
    )

    assert result.metrics["count"] == 1
    assert result.metrics["cost_total_usd"] == 0.004
    assert result.usage.samples == 1
    assert result.status == "ok"


async def test_expert_rejects_non_target_evidence(tmp_path: Path) -> None:
    evidence = EvidenceBundle(
        kind=InputKind.PASTE,
        pasted={"query": "q", "response": "a"},
        manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="requires target evidence"):
        await CostExpert(build_tools()).run(
            run_id="run-1",
            plan=cost_plan(),
            evidence=evidence,
        )
