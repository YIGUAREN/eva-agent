from datetime import UTC, datetime

from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import (
    Approval,
    DimensionResult,
    EvidenceRef,
    Finding,
    GraderRunOutput,
    GraderSampleOutcome,
    Usage,
)


def test_dimension_result_round_trips_json() -> None:
    result = DimensionResult(
        dimension=Dimension.QUALITY,
        status="ok",
        score=0.75,
        confidence=1.0,
        summary="平均分 0.75",
        metrics={"raw": [5, 3]},
        findings=[
            Finding(
                id="f1",
                severity="low",
                category="quality",
                title="低分样本",
                description="样本 2 得分较低",
                evidence=[EvidenceRef(kind="dataset_row", locator="2", excerpt="回答不完整")],
            )
        ],
        evidence=[EvidenceRef(kind="file", locator="dataset.csv")],
        usage=Usage(grader_calls=2, samples=2, duration_ms=50),
    )

    assert DimensionResult.model_validate(result.model_dump(mode="json")) == result


def test_approval_requires_plan_and_timestamp() -> None:
    from eva_agent.domain.budget import Budget
    from eva_agent.domain.plan import EvalPlan

    plan = EvalPlan(
        objective="o",
        evidence_summary="e",
        dimensions=[],
        skipped=[],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=10.0, max_target_requests=1),
    )
    approval = Approval(plan=plan, decided_at=datetime.now(UTC))

    assert Approval.model_validate(approval.model_dump(mode="json")) == approval


def test_grader_run_output_keeps_errors() -> None:
    output = GraderRunOutput(
        outcomes=[GraderSampleOutcome(grader="correctness", sample_index=0, error="boom")],
        duration_ms=12,
    )

    assert output.outcomes[0].error == "boom"
