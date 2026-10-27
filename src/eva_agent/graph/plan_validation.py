"""Validate a plan before it is approved and resumed."""

from __future__ import annotations

from eva_agent.domain.plan import Dimension, EvalPlan

IMPLEMENTED_DIMENSIONS = frozenset({Dimension.QUALITY})


class PlanValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def validate_plan(
    plan: EvalPlan,
    *,
    catalog_ids: set[str],
    original: EvalPlan | None = None,
    implemented: frozenset[Dimension] = IMPLEMENTED_DIMENSIONS,
) -> None:
    dimensions = [item.dimension for item in plan.dimensions]
    if len(set(dimensions)) != len(dimensions):
        raise PlanValidationError("dimensions", "duplicate dimension")

    enabled = [item for item in plan.dimensions if item.enabled]
    if not enabled:
        raise PlanValidationError("dimensions", "at least one dimension must be enabled")

    skipped = {item.dimension for item in plan.skipped}
    enabled_dimensions = {item.dimension for item in enabled}

    for item in plan.dimensions:
        path = f"dimensions[{item.dimension.value}]"
        if not item.enabled:
            continue
        if item.dimension not in implemented:
            raise PlanValidationError(
                f"{path}.enabled",
                f"dimension '{item.dimension.value}' is not implemented in phase 1",
            )
        if not item.graders:
            raise PlanValidationError(f"{path}.graders", "an enabled dimension needs at least one grader")
        names = [spec.name for spec in item.graders]
        if len(set(names)) != len(names):
            raise PlanValidationError(f"{path}.graders", "duplicate grader name")
        for index, spec in enumerate(item.graders):
            spec_path = f"{path}.graders[{index}]"
            if spec.level == "declarative":
                raise PlanValidationError(f"{spec_path}.level", "declarative graders are not implemented until phase 2")
            if spec.mode != "pointwise":
                raise PlanValidationError(f"{spec_path}.mode", "phase 1 only supports pointwise graders")
            if spec.level == "builtin" and (spec.builtin_id is None or spec.builtin_id not in catalog_ids):
                raise PlanValidationError(
                    f"{spec_path}.builtin_id",
                    f"unknown builtin_id '{spec.builtin_id}'",
                )
            if spec.level == "llm_rubric" and not (spec.rubric and spec.rubric.strip()):
                raise PlanValidationError(f"{spec_path}.rubric", "llm_rubric grader requires a rubric")

    if skipped & enabled_dimensions:
        raise PlanValidationError("skipped", "an enabled dimension must not also be skipped")

    for dimension in Dimension:
        if dimension not in enabled_dimensions and dimension not in skipped:
            raise PlanValidationError("skipped", f"missing skip reason for '{dimension.value}'")

    if original is not None:
        if plan.objective != original.objective:
            raise PlanValidationError("objective", "objective is immutable after planning")
        if plan.evidence_summary != original.evidence_summary:
            raise PlanValidationError("evidence_summary", "evidence_summary is immutable after planning")
