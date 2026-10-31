"""Recompute a packaged run offline and compare it with its recorded report."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan
from eva_agent.domain.result import DimensionResult, GraderRunOutput, GraderSampleOutcome
from eva_agent.graders_dsl.compiler import compile_grader
from eva_agent.graph.experts.cost import build_cost_record, summarize_cost
from eva_agent.graph.experts.latency import build_latency_record, is_successful, summarize_latency
from eva_agent.graph.experts.quality import aggregate_outcomes
from eva_agent.packages.grader_package import GraderPackage
from eva_agent.packages.manifest import PackageError
from eva_agent.pricing import ModelPrice
from eva_agent.security.passive import passive_specs, summarize_passive

COMPARED_FIELDS = (
    "status",
    "score",
    "confidence",
    "summary",
    "metrics",
    "findings",
    "error",
    "truncated_reason",
)


@dataclass(frozen=True)
class Difference:
    path: str
    recorded: Any
    replayed: Any


@dataclass(frozen=True)
class ReplayResult:
    matched: bool
    compared: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    differences: list[Difference] = field(default_factory=list)


def _plan_dimension(plan: EvalPlan, dimension: Dimension) -> DimensionPlan | None:
    for item in plan.dimensions:
        if item.dimension is dimension and item.enabled:
            return item
    return None


def _dimension_rows(package: GraderPackage, dimension: Dimension) -> list[dict[str, Any]]:
    """The rows the expert graded: that dimension's packaged limit, capped by what is packaged."""
    limit = package.inputs.dimension_limits.get(dimension.value, len(package.inputs.rows))
    return package.inputs.rows[:limit]


def _dimension_indexes(package: GraderPackage, dimension: Dimension) -> list[int | None]:
    return package.inputs.source_indexes[: len(_dimension_rows(package, dimension))]


def _recorded_output(dimension: Dimension, specs: list[GraderSpec], package: GraderPackage) -> GraderRunOutput:
    outcomes = [
        GraderSampleOutcome(
            grader=spec.name,
            sample_index=item.sample_index,
            score=item.score,
            raw_score=item.raw_score,
            # ``RecordedOutcome`` keeps an absent reason as ``None``; the
            # aggregated outcome model requires a string.
            reason=item.reason or "",
            error=item.error,
        )
        for spec in specs
        for item in package.snapshots.for_spec(dimension, spec.name)
    ]
    return GraderRunOutput(outcomes=sorted(outcomes, key=lambda item: (item.grader, item.sample_index)))


def _declarative_spec(dimension: DimensionPlan) -> GraderSpec:
    for spec in dimension.graders:
        if spec.level == "declarative":
            return spec
    raise PackageError(f"{dimension.dimension.value} has no declarative grader")


def _declarative_output(spec: GraderSpec, rows: list[dict[str, Any]]) -> GraderRunOutput:
    grader = compile_grader(spec)
    outcomes: list[GraderSampleOutcome] = []
    for index, record in enumerate(rows):
        score, checks = grader.evaluate(record)
        failed = [check.reason for check in checks if not check.passed]
        outcomes.append(
            GraderSampleOutcome(
                grader=spec.name,
                sample_index=index,
                score=score,
                raw_score=score,
                reason="; ".join(failed),
            )
        )
    return GraderRunOutput(outcomes=outcomes)


def _price_for(package: GraderPackage) -> ModelPrice:
    target = package.inputs.evidence.target
    if target is None:
        raise PackageError("cost replay requires a target")
    for price in package.inputs.prices:
        if price.provider == target.provider and price.model == target.model:
            return price
    raise PackageError(f"package has no price row for {target.provider}/{target.model}")


def _truncated_records(package: GraderPackage, dimension: DimensionPlan) -> list[Any]:
    limit = dimension.sample_size
    records = list(package.inputs.probe_records)
    return records[:limit] if limit else records


def _quality_values(package: GraderPackage) -> dict[str, Any]:
    dimension = _plan_dimension(package.plan, Dimension.QUALITY)
    if dimension is None:
        raise PackageError("quality is not enabled")
    rows = _dimension_rows(package, Dimension.QUALITY)
    if not rows:
        # The expert refuses to summarize zero rows, so the recorded entry
        # cannot be re-derived faithfully.
        raise PackageError("quality: the dimension has no inputs to replay")
    specs = [spec for spec in dimension.graders if spec.mode == "pointwise" and spec.level != "declarative"]
    normalized = aggregate_outcomes(
        _recorded_output(Dimension.QUALITY, specs, package),
        specs=specs,
        rows=rows,
        evidence=package.inputs.evidence.as_bundle(),
        source_indexes=_dimension_indexes(package, Dimension.QUALITY),
    )
    # ``summarize_quality`` records the pointwise grader count inside the
    # dimension metrics, so the report carries it and replay must re-derive it.
    metrics = {**normalized.metrics, "grader_count": len(specs)}
    return {
        "score": normalized.score,
        "confidence": normalized.confidence,
        "metrics": metrics,
    }


def _security_values(package: GraderPackage) -> dict[str, Any]:
    if _plan_dimension(package.plan, Dimension.SECURITY) is None:
        raise PackageError("security is not enabled")
    rows = _dimension_rows(package, Dimension.SECURITY)
    if not rows:
        # The expert fails the dimension instead of summarizing zero rows.
        raise PackageError("security: the dimension has no inputs to replay")
    specs = passive_specs(rows)
    normalized = summarize_passive(
        _recorded_output(Dimension.SECURITY, specs, package),
        specs=specs,
        rows=rows,
        evidence=package.inputs.evidence.as_bundle(),
        source_indexes=_dimension_indexes(package, Dimension.SECURITY),
    )
    return {
        "score": normalized.score,
        "confidence": normalized.confidence,
        "metrics": normalized.metrics,
    }


def _cost_result(package: GraderPackage) -> DimensionResult:
    dimension = _plan_dimension(package.plan, Dimension.COST)
    if dimension is None:
        raise PackageError("cost is not enabled")
    spec = _declarative_spec(dimension)
    price = _price_for(package)
    records = _truncated_records(package, dimension)
    if not records:
        # The expert raises on an empty probe file instead of summarizing it.
        raise PackageError("cost: the dimension has no inputs to replay")
    rows: list[dict[str, Any]] = []
    costs: list[Decimal] = []
    for record in records:
        if record.input_tokens is None or record.output_tokens is None:
            continue
        cost = price.cost_usd(input_tokens=record.input_tokens, output_tokens=record.output_tokens)
        rows.append(build_cost_record(record, cost))
        costs.append(cost)
    return summarize_cost(
        records=records,
        rows=rows,
        costs=costs,
        outcomes=[_declarative_output(spec, rows)],
        specs=[spec],
        sample_size=dimension.sample_size or len(records),
    )


def _latency_result(package: GraderPackage) -> DimensionResult:
    dimension = _plan_dimension(package.plan, Dimension.LATENCY)
    if dimension is None:
        raise PackageError("latency is not enabled")
    spec = _declarative_spec(dimension)
    records = _truncated_records(package, dimension)
    if not records:
        # The expert raises on an empty probe file instead of summarizing it.
        raise PackageError("latency: the dimension has no inputs to replay")
    # The latency expert grades only successful requests, exactly as here.
    rows = [build_latency_record(record) for record in records if is_successful(record)]
    return summarize_latency(
        records=records,
        outcomes=[_declarative_output(spec, rows)],
        specs=[spec],
        sample_size=dimension.sample_size or len(records),
    )


def _pair(path: str, recorded: Any, replayed: Any, differences: list[Difference]) -> None:
    if recorded != replayed:
        differences.append(Difference(path, recorded, replayed))


def replay_package(package: GraderPackage) -> ReplayResult:
    if not package.manifest.replayable:
        blockers = "; ".join(package.manifest.replay_blockers)
        raise PackageError(f"package is not replayable: {blockers}")
    recorded = {item["dimension"]: item for item in package.report.get("dimensions") or []}
    compared: list[str] = []
    skipped: list[str] = []
    differences: list[Difference] = []
    for dimension in package.plan.dimensions:
        if not dimension.enabled:
            continue
        name = dimension.dimension.value
        entry = recorded.get(name)
        if entry is None:
            skipped.append(name)
            continue
        if dimension.dimension is Dimension.QUALITY:
            values = _quality_values(package)
            for key, value in values.items():
                _pair(f"dimensions.{name}.{key}", entry.get(key), value, differences)
        elif dimension.dimension is Dimension.SECURITY:
            values = _security_values(package)
            metrics = entry.get("metrics") or {}
            active = metrics.get("active") or {}
            _pair(f"dimensions.{name}.metrics.passive", metrics.get("passive"), values["metrics"], differences)
            if metrics.get("score_cap") is None and active.get("status") == "not_applicable":
                _pair(f"dimensions.{name}.score", entry.get("score"), values["score"], differences)
                _pair(
                    f"dimensions.{name}.confidence",
                    entry.get("confidence"),
                    values["confidence"],
                    differences,
                )
        elif dimension.dimension is Dimension.COST:
            payload = _cost_result(package).model_dump(mode="json")
            for field_name in COMPARED_FIELDS:
                _pair(f"dimensions.{name}.{field_name}", entry.get(field_name), payload[field_name], differences)
        elif dimension.dimension is Dimension.LATENCY:
            payload = _latency_result(package).model_dump(mode="json")
            for field_name in COMPARED_FIELDS:
                _pair(f"dimensions.{name}.{field_name}", entry.get(field_name), payload[field_name], differences)
        else:
            skipped.append(name)
            continue
        compared.append(name)
    return ReplayResult(
        matched=not differences,
        compared=compared,
        skipped=skipped,
        differences=differences,
    )
