"""Collect the recorded grader outcomes a package replays instead of calling a model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from eva_agent.adapters.openjudge.cache import GradingCache, cache_key
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.security.passive import passive_specs


class RecordedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    spec_name: str
    sample_index: int = Field(ge=0)
    score: float | None = None
    raw_score: float | None = None
    reason: str | None = None
    error: str | None = None


class SnapshotSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcomes: list[RecordedOutcome] = Field(default_factory=list)

    def for_spec(self, spec_name: str) -> list[RecordedOutcome]:
        return [item for item in self.outcomes if item.spec_name == spec_name]


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
) -> tuple[SnapshotSet, list[str]]:
    """Read every cached outcome for every model-backed spec; list what is missing."""
    cache = GradingCache(Path(data_dir), run_id)
    outcomes: list[RecordedOutcome] = []
    blockers: list[str] = []
    for dimension_plan in dimensions:
        for spec in _snapshot_specs(dimension_plan, rows):
            recorded = 0
            for index, row in enumerate(rows):
                key = cache_key(spec, row)
                hit = cache.read(spec.name, key, index)
                if hit is None:
                    continue
                recorded += 1
                outcomes.append(
                    RecordedOutcome(
                        spec_name=spec.name,
                        sample_index=index,
                        score=hit.score,
                        raw_score=hit.raw_score,
                        reason=hit.reason,
                        error=hit.error,
                    )
                )
            if recorded != len(rows):
                blockers.append(
                    f"{dimension_plan.dimension.value}/{spec.name}: "
                    f"{recorded} of {len(rows)} samples were recorded"
                )
    return SnapshotSet(outcomes=outcomes), blockers
