"""Validate a plan before it is approved and resumed."""

from __future__ import annotations

from pydantic import ValidationError

from eva_agent.domain.plan import Dimension, EvalPlan
from eva_agent.domain.run import InputKind
from eva_agent.graders_dsl.schema import DeclarativeSpecError, validate_declarative_spec

IMPLEMENTED_DIMENSIONS = frozenset({Dimension.QUALITY, Dimension.LATENCY, Dimension.COST})
TARGET_DIMENSIONS = frozenset({Dimension.LATENCY, Dimension.COST})


class PlanValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def validate_plan(
    plan: EvalPlan,
    *,
    catalog_ids: set[str],
    input_kind: InputKind = InputKind.PASTE,
    max_target_requests: int | None = None,
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
                f"dimension '{item.dimension.value}' is not implemented in phase 2",
            )
        if item.dimension in TARGET_DIMENSIONS:
            if input_kind is not InputKind.TARGET:
                raise PlanValidationError(
                    f"{path}.enabled",
                    f"dimension '{item.dimension.value}' requires target input",
                )
            if item.sample_size is None:
                raise PlanValidationError(f"{path}.sample_size", "target dimensions require sample_size")
        elif input_kind is InputKind.TARGET and item.dimension is Dimension.QUALITY:
            raise PlanValidationError(
                f"{path}.enabled",
                "quality on target input is not implemented in phase 2",
            )

        if not item.graders:
            raise PlanValidationError(f"{path}.graders", "an enabled dimension needs at least one grader")
        names = [spec.name for spec in item.graders]
        if len(set(names)) != len(names):
            raise PlanValidationError(f"{path}.graders", "duplicate grader name")

        for index, spec in enumerate(item.graders):
            spec_path = f"{path}.graders[{index}]"
            if spec.mode != "pointwise":
                raise PlanValidationError(f"{spec_path}.mode", "only pointwise graders are supported")
            if item.dimension in TARGET_DIMENSIONS:
                if spec.level != "declarative":
                    raise PlanValidationError(
                        f"{spec_path}.level",
                        "target dimensions require exactly one declarative grader",
                    )
            elif spec.level == "declarative":
                raise PlanValidationError(
                    f"{spec_path}.level",
                    "declarative graders are only valid for target dimensions",
                )
            if spec.level == "builtin" and (spec.builtin_id is None or spec.builtin_id not in catalog_ids):
                raise PlanValidationError(
                    f"{spec_path}.builtin_id",
                    f"unknown builtin_id '{spec.builtin_id}'",
                )
            if spec.level == "llm_rubric" and not (spec.rubric and spec.rubric.strip()):
                raise PlanValidationError(f"{spec_path}.rubric", "llm_rubric grader requires a rubric")

        if item.dimension in TARGET_DIMENSIONS:
            declarative = [spec for spec in item.graders if spec.level == "declarative"]
            if len(declarative) != 1:
                raise PlanValidationError(
                    f"{path}.graders",
                    "target dimensions require exactly one declarative grader",
                )
            try:
                validate_declarative_spec(declarative[0])
            except DeclarativeSpecError as exc:
                raise PlanValidationError(f"{path}.graders[0]", str(exc)) from None

    if skipped & enabled_dimensions:
        raise PlanValidationError("skipped", "an enabled dimension must not also be skipped")

    for dimension in Dimension:
        if dimension not in enabled_dimensions and dimension not in skipped:
            raise PlanValidationError("skipped", f"missing skip reason for '{dimension.value}'")

    target_enabled = [item for item in enabled if item.dimension in TARGET_DIMENSIONS]
    if target_enabled:
        probe_count = max(item.sample_size or 0 for item in target_enabled)
        limit = max_target_requests if max_target_requests is not None else plan.budget.max_target_requests
        if probe_count > limit:
            raise PlanValidationError(
                "dimensions",
                f"probe count {probe_count} exceeds max_target_requests {limit}",
            )

    if original is not None:
        if plan.objective != original.objective:
            raise PlanValidationError("objective", "objective is immutable after planning")
        if plan.evidence_summary != original.evidence_summary:
            raise PlanValidationError("evidence_summary", "evidence_summary is immutable after planning")


def parse_plan_edit(
    payload: object,
    *,
    catalog_ids: set[str],
    original: EvalPlan,
    input_kind: InputKind = InputKind.PASTE,
) -> EvalPlan:
    try:
        candidate = EvalPlan.model_validate(payload)
    except ValidationError as exc:
        raise PlanValidationError("plan", str(exc)) from exc
    validate_plan(candidate, catalog_ids=catalog_ids, input_kind=input_kind, original=original)
    return candidate
