import pytest

from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import CheckSpec, GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.run import InputKind
from eva_agent.graph.plan_validation import PlanValidationError, parse_plan_edit, validate_plan


def declarative(name: str, field: str, threshold: float) -> GraderSpec:
    return GraderSpec(
        level="declarative",
        name=name,
        input_map={field: f"$.{field}"},
        checks=[CheckSpec(type="numeric_threshold", field=field, op="<=", value=threshold)],
    )


def target_plan(*, sample_size: int = 3, max_requests: int = 5) -> EvalPlan:
    return EvalPlan(
        objective="评估目标延迟与成本",
        evidence_summary="target demo",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.LATENCY,
                enabled=True,
                sample_size=sample_size,
                graders=[declarative("latency_budget", "latency_ms", 3000)],
            ),
            DimensionPlan(
                dimension=Dimension.COST,
                enabled=True,
                sample_size=sample_size,
                graders=[declarative("cost_budget", "cost_usd", 0.01)],
            ),
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.QUALITY, reason="阶段 2 不在目标响应上评质量"),
            SkippedDimension(dimension=Dimension.SECURITY, reason="阶段 3 实现"),
        ],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=max_requests),
    )


def test_target_plan_passes():
    validate_plan(target_plan(), catalog_ids=set(), input_kind=InputKind.TARGET)


def test_target_dimension_requires_target_input():
    with pytest.raises(PlanValidationError, match="requires target input"):
        validate_plan(target_plan(), catalog_ids=set(), input_kind=InputKind.PASTE)


def test_target_dimension_requires_sample_size():
    plan = target_plan().model_copy(
        update={
            "dimensions": [
                target_plan().dimensions[0].model_copy(update={"sample_size": None}),
                target_plan().dimensions[1],
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="sample_size"):
        validate_plan(plan, catalog_ids=set(), input_kind=InputKind.TARGET)


def test_target_dimension_requires_exactly_one_declarative_grader():
    plan = target_plan().model_copy(
        update={
            "dimensions": [
                target_plan()
                .dimensions[0]
                .model_copy(
                    update={
                        "graders": [
                            declarative("a", "latency_ms", 1),
                            declarative("b", "latency_ms", 2),
                        ]
                    }
                ),
                target_plan().dimensions[1],
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="exactly one declarative grader"):
        validate_plan(plan, catalog_ids=set(), input_kind=InputKind.TARGET)


def test_target_dimension_rejects_builtin_grader():
    plan = target_plan().model_copy(
        update={
            "dimensions": [
                target_plan()
                .dimensions[0]
                .model_copy(update={"graders": [GraderSpec(level="builtin", name="x", builtin_id="relevance")]}),
                target_plan().dimensions[1],
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="exactly one declarative grader"):
        validate_plan(plan, catalog_ids=set(), input_kind=InputKind.TARGET)


def test_invalid_l2_spec_rejected():
    bad = GraderSpec(
        level="declarative",
        name="bad",
        input_map={"latency_ms": "$.latency_ms"},
        checks=[CheckSpec(type="json_parse", field="missing")],
    )
    plan = target_plan().model_copy(
        update={
            "dimensions": [
                target_plan().dimensions[0].model_copy(update={"graders": [bad]}),
                target_plan().dimensions[1],
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="not declared"):
        validate_plan(plan, catalog_ids=set(), input_kind=InputKind.TARGET)


def test_probe_count_must_fit_budget():
    with pytest.raises(PlanValidationError, match="max_target_requests"):
        validate_plan(target_plan(sample_size=6, max_requests=5), catalog_ids=set(), input_kind=InputKind.TARGET)


def test_explicit_max_target_requests_overrides_plan_budget():
    validate_plan(
        target_plan(sample_size=6, max_requests=5),
        catalog_ids=set(),
        input_kind=InputKind.TARGET,
        max_target_requests=6,
    )


def test_quality_is_not_implemented_on_target():
    plan = target_plan().model_copy(
        update={
            "dimensions": [
                DimensionPlan(
                    dimension=Dimension.QUALITY,
                    enabled=True,
                    graders=[GraderSpec(level="builtin", name="c", builtin_id="correctness")],
                )
            ],
            "skipped": [
                SkippedDimension(dimension=Dimension.LATENCY, reason="x"),
                SkippedDimension(dimension=Dimension.COST, reason="x"),
                SkippedDimension(dimension=Dimension.SECURITY, reason="x"),
            ],
        }
    )

    with pytest.raises(PlanValidationError, match="quality"):
        validate_plan(plan, catalog_ids={"correctness"}, input_kind=InputKind.TARGET)


def test_parse_plan_edit_uses_input_kind():
    plan = target_plan()
    payload = plan.model_dump(mode="json")

    parse_plan_edit(payload, catalog_ids=set(), original=plan, input_kind=InputKind.TARGET)

    with pytest.raises(PlanValidationError, match="requires target input"):
        parse_plan_edit(payload, catalog_ids=set(), original=plan)
