import pytest

from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import CheckSpec, GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.run import InputKind
from eva_agent.graph.plan_validation import PlanValidationError, validate_plan


def dimension(dim: Dimension, *, graders=None, sample_size=2) -> DimensionPlan:
    return DimensionPlan(dimension=dim, enabled=True, sample_size=sample_size, graders=graders or [])


def declarative(name: str, field: str) -> GraderSpec:
    return GraderSpec(
        level="declarative",
        name=name,
        input_map={field: f"$.{field}", "response": "$.response"},
        checks=[CheckSpec(type="numeric_threshold", field=field, op="<=", value=1000)],
    )


def plan(*dimensions: DimensionPlan) -> EvalPlan:
    enabled = {item.dimension for item in dimensions}
    return EvalPlan(
        objective="阶段 3",
        evidence_summary="target",
        dimensions=list(dimensions),
        skipped=[SkippedDimension(dimension=item, reason="not selected") for item in Dimension if item not in enabled],
        budget=Budget(max_cost_usd=1, max_wall_clock_s=60, max_target_requests=20),
    )


def test_security_is_implemented_with_no_graders() -> None:
    validate_plan(plan(dimension(Dimension.SECURITY)), catalog_ids=set(), input_kind=InputKind.TARGET)


def test_security_passive_is_allowed_on_paste() -> None:
    validate_plan(plan(dimension(Dimension.SECURITY)), catalog_ids=set(), input_kind=InputKind.PASTE)


def test_security_graders_are_rejected() -> None:
    bad = dimension(
        Dimension.SECURITY,
        graders=[GraderSpec(level="builtin", name="harmfulness", builtin_id="harmfulness")],
    )
    with pytest.raises(PlanValidationError, match="graders are fixed"):
        validate_plan(plan(bad), catalog_ids={"harmfulness"}, input_kind=InputKind.TARGET)


def test_target_quality_accepts_relevance_and_rubric() -> None:
    validate_plan(
        plan(
            dimension(
                Dimension.QUALITY,
                graders=[
                    GraderSpec(level="builtin", name="relevance", builtin_id="relevance"),
                    GraderSpec(level="llm_rubric", name="quality_rubric", rubric="- good"),
                ],
            )
        ),
        catalog_ids={"relevance"},
        input_kind=InputKind.TARGET,
    )


def test_target_quality_rejects_reference_or_unsupported_grader() -> None:
    with pytest.raises(PlanValidationError, match="only relevance"):
        validate_plan(
            plan(
                dimension(
                    Dimension.QUALITY,
                    graders=[GraderSpec(level="builtin", name="correctness", builtin_id="correctness")],
                )
            ),
            catalog_ids={"correctness"},
            input_kind=InputKind.TARGET,
        )


def test_target_quality_rejects_declarative_grader() -> None:
    bad = GraderSpec(
        level="declarative",
        name="q",
        input_map={"latency_ms": "$.latency_ms"},
        checks=[CheckSpec(type="numeric_threshold", field="latency_ms", op="<=", value=1)],
    )
    with pytest.raises(PlanValidationError, match="declarative"):
        validate_plan(
            plan(dimension(Dimension.QUALITY, graders=[bad])),
            catalog_ids=set(),
            input_kind=InputKind.TARGET,
        )


def test_all_four_dimensions_can_be_enabled() -> None:
    validate_plan(
        plan(
            dimension(
                Dimension.QUALITY, graders=[GraderSpec(level="builtin", name="relevance", builtin_id="relevance")]
            ),
            dimension(Dimension.LATENCY, graders=[declarative("latency_budget", "latency_ms")]),
            dimension(Dimension.COST, graders=[declarative("cost_budget", "cost_usd")]),
            dimension(Dimension.SECURITY),
        ),
        catalog_ids={"relevance"},
        input_kind=InputKind.TARGET,
    )
