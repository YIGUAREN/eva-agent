"""Convert OpenJudge results into normalized per-sample outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from openjudge.graders.schema import GraderError, GraderRank, GraderScore

from eva_agent.adapters.openjudge.catalog import GraderEntry
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderSampleOutcome


def _normalized(raw: float, score_min: float, score_max: float) -> float | None:
    span = score_max - score_min
    if span <= 0:
        return None
    return min(1.0, max(0.0, (raw - score_min) / span))


def normalize_results(
    results: Mapping[str, list[Any]],
    specs: list[GraderSpec],
    catalog: Mapping[str, GraderEntry],
    sample_count: int,
) -> list[GraderSampleOutcome]:
    outcomes: list[GraderSampleOutcome] = []
    for spec in specs:
        entry = catalog.get(spec.builtin_id or "") if spec.level == "builtin" else None
        if spec.level == "builtin" and entry is None:
            message = f"unknown grader '{spec.builtin_id}'"
            outcomes.extend(
                GraderSampleOutcome(grader=spec.name, sample_index=index, error=message)
                for index in range(sample_count)
            )
            continue

        score_min = entry.score_min if entry is not None else 0.0
        score_max = entry.score_max if entry is not None else 1.0
        if score_max <= score_min:
            message = f"invalid score range for '{spec.name}'"
            outcomes.extend(
                GraderSampleOutcome(grader=spec.name, sample_index=index, error=message)
                for index in range(sample_count)
            )
            continue

        raw_results = results.get(spec.name, [])
        for index in range(sample_count):
            if index >= len(raw_results):
                outcomes.append(GraderSampleOutcome(grader=spec.name, sample_index=index, error="missing result"))
                continue
            result = raw_results[index]
            if isinstance(result, GraderError):
                outcomes.append(
                    GraderSampleOutcome(
                        grader=spec.name,
                        sample_index=index,
                        reason=result.reason,
                        error=result.error,
                    )
                )
            elif isinstance(result, GraderScore):
                raw_score = float(result.score)
                if not math.isfinite(raw_score):
                    outcomes.append(
                        GraderSampleOutcome(
                            grader=spec.name,
                            sample_index=index,
                            reason=result.reason,
                            error=f"non-finite score: {raw_score!r}",
                        )
                    )
                    continue
                outcomes.append(
                    GraderSampleOutcome(
                        grader=spec.name,
                        sample_index=index,
                        score=_normalized(raw_score, score_min, score_max),
                        raw_score=raw_score,
                        reason=result.reason,
                    )
                )
            elif isinstance(result, GraderRank):
                outcomes.append(
                    GraderSampleOutcome(
                        grader=spec.name,
                        sample_index=index,
                        reason=result.reason,
                        error="listwise results are not supported in phase 1",
                    )
                )
            else:
                outcomes.append(
                    GraderSampleOutcome(
                        grader=spec.name,
                        sample_index=index,
                        error=f"unsupported grader result: {type(result).__name__}",
                    )
                )
    return outcomes
