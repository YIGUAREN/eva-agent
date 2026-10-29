"""Shared aggregation of declarative grader outcomes for the experts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any

from eva_agent.domain.result import GraderRunOutput


@dataclass(frozen=True)
class GraderAggregate:
    """The score, confidence, and shared metrics derived from grader outcomes."""

    score: float | None
    confidence: float
    scored: int
    calls: int
    grader_scores: dict[str, float]
    errors: list[str]

    def metrics(self, sample_size: int) -> dict[str, Any]:
        """Return the metric keys every expert reports for its graders."""
        return {
            "grader_scores": self.grader_scores,
            "scored_sample_count": self.scored,
            "sample_size": sample_size,
            "errors": self.errors,
        }


def aggregate_graders(outcomes: list[GraderRunOutput], sample_size: int) -> GraderAggregate:
    """Fold grader outcomes into a per-sample mean score plus shared metrics."""
    by_sample: dict[int, list[float]] = defaultdict(list)
    by_grader: dict[str, list[float]] = defaultdict(list)
    errors: set[str] = set()
    scored = 0
    calls = 0
    for output in outcomes:
        for outcome in output.outcomes:
            calls += 1
            if outcome.score is None:
                if outcome.error:
                    errors.add(outcome.error)
                continue
            scored += 1
            by_sample[outcome.sample_index].append(outcome.score)
            by_grader[outcome.grader].append(outcome.score)

    sample_scores = {index: mean(values) for index, values in sorted(by_sample.items())}
    return GraderAggregate(
        score=mean(sample_scores.values()) if sample_scores else None,
        confidence=min(1.0, scored / sample_size) if sample_size else 0.0,
        scored=scored,
        calls=calls,
        grader_scores={name: round(mean(values), 4) for name, values in by_grader.items()},
        errors=sorted(errors),
    )
