import pytest
from pydantic import ValidationError

from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import GraderSpec, RubricSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension


def sample_budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=600.0, max_target_requests=10)


def test_plan_requires_objective() -> None:
    with pytest.raises(ValidationError):
        EvalPlan(objective="", evidence_summary="none", dimensions=[], skipped=[], budget=sample_budget())


def test_plan_round_trips_json() -> None:
    plan = EvalPlan(
        objective="评估回答质量",
        evidence_summary="1 条问答",
        dimensions=[DimensionPlan(dimension=Dimension.QUALITY, enabled=True, sample_size=5, notes="")],
        skipped=[SkippedDimension(dimension=Dimension.COST, reason="缺少 token 数据")],
        budget=sample_budget(),
    )

    assert EvalPlan.model_validate(plan.model_dump(mode="json")) == plan


def test_sample_size_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        DimensionPlan(dimension=Dimension.QUALITY, enabled=True, sample_size=0, notes="")


def test_plan_with_graders_round_trips_json() -> None:
    plan = EvalPlan(
        objective="评估回答质量",
        evidence_summary="1 条问答",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.QUALITY,
                enabled=True,
                sample_size=5,
                graders=[GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")],
                rubrics=[RubricSpec(name="准确性", description="与参考答案一致")],
            )
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.LATENCY, reason="阶段 1 未实现"),
            SkippedDimension(dimension=Dimension.COST, reason="阶段 1 未实现"),
            SkippedDimension(dimension=Dimension.SECURITY, reason="阶段 1 未实现"),
        ],
        budget=sample_budget(),
    )

    assert EvalPlan.model_validate(plan.model_dump(mode="json")) == plan
