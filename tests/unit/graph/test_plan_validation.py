import pytest

from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import CheckSpec, GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.graph.plan_validation import PlanValidationError, validate_plan


def budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1)


def skipped_all_but_quality() -> list[SkippedDimension]:
    return [
        SkippedDimension(dimension=Dimension.LATENCY, reason="阶段 1 未实现"),
        SkippedDimension(dimension=Dimension.COST, reason="阶段 1 未实现"),
        SkippedDimension(dimension=Dimension.SECURITY, reason="阶段 1 未实现"),
    ]


def valid_plan() -> EvalPlan:
    return EvalPlan(
        objective="评估回答质量",
        evidence_summary="1 条样本",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.QUALITY,
                enabled=True,
                sample_size=5,
                graders=[GraderSpec(level="builtin", name="correctness", builtin_id="correctness")],
            )
        ],
        skipped=skipped_all_but_quality(),
        budget=budget(),
    )


def test_valid_plan_passes() -> None:
    validate_plan(valid_plan(), catalog_ids={"correctness"})


def test_objective_change_is_rejected() -> None:
    original = valid_plan()
    changed = original.model_copy(update={"objective": "换一个目标"})

    with pytest.raises(PlanValidationError, match="objective"):
        validate_plan(changed, catalog_ids={"correctness"}, original=original)


def test_evidence_summary_change_is_rejected() -> None:
    original = valid_plan()
    changed = original.model_copy(update={"evidence_summary": "换一个摘要"})

    with pytest.raises(PlanValidationError, match="evidence_summary"):
        validate_plan(changed, catalog_ids={"correctness"}, original=original)


def test_unimplemented_dimension_cannot_be_enabled() -> None:
    plan = valid_plan().model_copy(
        update={
            "dimensions": [
                *valid_plan().dimensions,
                DimensionPlan(dimension=Dimension.SECURITY, enabled=True),
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="not implemented"):
        validate_plan(
            plan,
            catalog_ids={"correctness"},
            implemented=frozenset({Dimension.QUALITY}),
        )


def test_enabled_dimension_requires_a_grader() -> None:
    plan = valid_plan().model_copy(
        update={"dimensions": [DimensionPlan(dimension=Dimension.QUALITY, enabled=True, graders=[])]}
    )

    with pytest.raises(PlanValidationError, match="at least one grader"):
        validate_plan(plan, catalog_ids={"correctness"})


def test_unknown_builtin_id_is_rejected() -> None:
    plan = valid_plan().model_copy(
        update={
            "dimensions": [
                DimensionPlan(
                    dimension=Dimension.QUALITY,
                    enabled=True,
                    graders=[GraderSpec(level="builtin", name="x", builtin_id="x")],
                )
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="unknown builtin_id"):
        validate_plan(plan, catalog_ids={"correctness"})


def test_declarative_grader_is_rejected_on_quality() -> None:
    spec = GraderSpec(
        level="declarative",
        name="x",
        input_map={"response": "$.response"},
        checks=[CheckSpec(type="json_parse", field="response")],
    )
    plan = valid_plan().model_copy(
        update={
            "dimensions": [
                DimensionPlan(
                    dimension=Dimension.QUALITY,
                    enabled=True,
                    graders=[spec],
                )
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="only valid for latency and cost"):
        validate_plan(plan, catalog_ids={"correctness"})


def test_llm_rubric_requires_text() -> None:
    plan = valid_plan().model_copy(
        update={
            "dimensions": [
                DimensionPlan(
                    dimension=Dimension.QUALITY,
                    enabled=True,
                    graders=[GraderSpec(level="llm_rubric", name="helpfulness")],
                )
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="rubric"):
        validate_plan(plan, catalog_ids={"correctness"})


def test_listwise_grader_is_rejected() -> None:
    plan = valid_plan().model_copy(
        update={
            "dimensions": [
                DimensionPlan(
                    dimension=Dimension.QUALITY,
                    enabled=True,
                    graders=[
                        GraderSpec(
                            level="builtin",
                            name="correctness",
                            mode="listwise",
                            builtin_id="correctness",
                        )
                    ],
                )
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="pointwise"):
        validate_plan(plan, catalog_ids={"correctness"})


def test_missing_skip_reason_is_rejected() -> None:
    plan = valid_plan().model_copy(update={"skipped": skipped_all_but_quality()[:2]})

    with pytest.raises(PlanValidationError, match="missing skip reason"):
        validate_plan(plan, catalog_ids={"correctness"})


def test_no_enabled_dimension_is_rejected() -> None:
    plan = valid_plan().model_copy(
        update={
            "dimensions": [DimensionPlan(dimension=Dimension.QUALITY, enabled=False)],
        }
    )

    with pytest.raises(PlanValidationError, match="at least one dimension"):
        validate_plan(plan, catalog_ids={"correctness"})


def test_duplicate_dimension_is_rejected() -> None:
    plan = valid_plan().model_copy(
        update={
            "dimensions": [*valid_plan().dimensions, *valid_plan().dimensions],
        }
    )

    with pytest.raises(PlanValidationError, match="duplicate"):
        validate_plan(plan, catalog_ids={"correctness"})


def test_duplicate_grader_name_is_rejected() -> None:
    spec = GraderSpec(level="builtin", name="correctness", builtin_id="correctness")
    plan = valid_plan().model_copy(
        update={
            "dimensions": [
                DimensionPlan(
                    dimension=Dimension.QUALITY,
                    enabled=True,
                    graders=[spec, spec],
                )
            ]
        }
    )

    with pytest.raises(PlanValidationError, match="duplicate grader name"):
        validate_plan(plan, catalog_ids={"correctness"})
