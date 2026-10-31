from decimal import Decimal

import pytest

from eva_agent.budget.estimator import BudgetPricingError, CostEstimator
from eva_agent.budget.finalize import estimate_finalize, finalize_input
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan
from eva_agent.domain.probe import ProbeSummary
from eva_agent.domain.result import DimensionResult, Finding, MergedResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.pricing import ModelPrice, PricingTable


def model() -> ModelConfig:
    return ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")


def plan() -> EvalPlan:
    return EvalPlan(
        objective="评估回答",
        evidence_summary="一段证据",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.QUALITY,
                enabled=True,
                sample_size=2,
                graders=[GraderSpec(level="builtin", name="relevance", builtin_id="relevance")],
            )
        ],
        skipped=[],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=2),
    )


def estimator() -> CostEstimator:
    return CostEstimator(
        PricingTable(
            [
                ModelPrice(
                    provider="anthropic",
                    model="demo-echo",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                )
            ]
        )
    )


def merged() -> MergedResult:
    return MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            DimensionResult(
                dimension=Dimension.QUALITY,
                status="ok",
                score=0.9,
                confidence=0.8,
                summary="distinctive-merged-summary-marker",
                findings=[
                    Finding(
                        id="finding-1",
                        severity="low",
                        category="quality",
                        title="distinctive-finding-title-marker",
                        description="a finding description",
                    )
                ],
                usage=Usage(samples=1),
            )
        ],
        skipped=[],
    )


def probe() -> ProbeSummary:
    return ProbeSummary(path="probe/distinctive-probe-path.jsonl", total=2, succeeded=1, failed=1)


def test_finalize_estimate_is_positive_and_reads_the_plan() -> None:
    text = finalize_input(plan=plan(), merged=None, probe=None)
    assert "评估回答" in text
    estimate = estimate_finalize(estimator(), model(), plan=plan())
    assert isinstance(estimate, Decimal) and estimate > 0


def test_merged_results_and_probe_reach_the_estimate() -> None:
    text = finalize_input(plan=plan(), merged=merged(), probe=probe())
    assert "distinctive-merged-summary-marker" in text
    assert "distinctive-finding-title-marker" in text
    assert "distinctive-probe-path" in text

    without_tail = estimate_finalize(estimator(), model(), plan=plan())
    with_tail = estimate_finalize(estimator(), model(), plan=plan(), merged=merged(), probe=probe())
    assert with_tail > without_tail


def test_a_missing_plan_needs_no_finalize_budget() -> None:
    assert estimate_finalize(estimator(), model(), plan=None) == Decimal("0")


def test_unknown_model_fails_closed() -> None:
    other = ModelConfig(provider="anthropic", model="missing", api_key_env="DEMO_KEY")
    with pytest.raises(BudgetPricingError, match="pricing.yaml"):
        estimate_finalize(estimator(), other, plan=plan())
