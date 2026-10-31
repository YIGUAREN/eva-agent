"""Collect the recorded grader outcomes a package replays instead of calling a model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from eva_agent.adapters.openjudge.cache import GradingCache, cache_key
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.packages.manifest import PackageError
from eva_agent.security.passive import passive_specs


class RecordedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: Dimension
    spec_name: str
    sample_index: int = Field(ge=0)
    score: float | None = None
    raw_score: float | None = None
    reason: str | None = None
    error: str | None = None


class SnapshotSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcomes: list[RecordedOutcome] = Field(default_factory=list)

    def for_spec(self, dimension: Dimension, spec_name: str) -> list[RecordedOutcome]:
        """Outcomes for one dimension's spec; equal names in other dimensions never merge."""
        return [
            item for item in self.outcomes if item.dimension == dimension and item.spec_name == spec_name
        ]


def _snapshot_specs(plan_dimension: DimensionPlan, rows: list[dict[str, Any]]) -> list[GraderSpec]:
    """The model-backed specs whose outcomes the expert would have asked the cache for.

    Quality grades its pointwise plan graders; security ignores the plan and
    derives its passive graders from the rows, so the snapshot must do the same.
    Declarative graders run locally at replay time and never need a snapshot.
    """
    if plan_dimension.dimension is Dimension.SECURITY:
        return passive_specs(rows)
    if plan_dimension.dimension is Dimension.QUALITY:
        return [
            spec
            for spec in plan_dimension.graders
            if spec.mode == "pointwise" and spec.level != "declarative"
        ]
    return [spec for spec in plan_dimension.graders if spec.level != "declarative"]


def collect_snapshots(
    *,
    data_dir: str | Path,
    run_id: str,
    dimensions: Iterable[DimensionPlan],
    rows: list[dict[str, Any]],
    dimension_limits: Mapping[str, int],
) -> tuple[SnapshotSet, list[str]]:
    """Read every cached outcome for every model-backed spec; list what is missing.

    ``dimension_limits`` holds the effective sample limit each enabled expert
    used, so every dimension reads exactly the row subset the expert graded.
    """
    cache = GradingCache(Path(data_dir), run_id)
    outcomes: list[RecordedOutcome] = []
    blockers: list[str] = []
    for dimension_plan in dimensions:
        dimension = dimension_plan.dimension
        limit = dimension_limits.get(dimension.value)
        if limit is None:
            raise PackageError(f"no sample limit was recorded for dimension '{dimension.value}'")
        dimension_rows = rows[:limit]
        for spec in _snapshot_specs(dimension_plan, dimension_rows):
            recorded = 0
            errored = 0
            for index, row in enumerate(dimension_rows):
                key = cache_key(spec, row)
                hit = cache.read(spec.name, key, index)
                if hit is None:
                    continue
                recorded += 1
                if hit.error is not None:
                    errored += 1
                outcomes.append(
                    RecordedOutcome(
                        dimension=dimension,
                        spec_name=spec.name,
                        sample_index=index,
                        score=hit.score,
                        raw_score=hit.raw_score,
                        reason=hit.reason,
                        error=hit.error,
                    )
                )
            if errored:
                blockers.append(
                    f"{dimension.value}/{spec.name}: "
                    f"{errored} of {len(dimension_rows)} recorded samples errored"
                )
            if recorded != len(dimension_rows):
                blockers.append(
                    f"{dimension.value}/{spec.name}: "
                    f"{recorded} of {len(dimension_rows)} samples were recorded"
                )
    return SnapshotSet(outcomes=outcomes), blockers
