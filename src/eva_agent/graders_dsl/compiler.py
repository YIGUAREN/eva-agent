"""Compile a validated declarative spec into a plain-Python grader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eva_agent.domain.grader import GraderSpec
from eva_agent.graders_dsl.checks import MISSING, run_check
from eva_agent.graders_dsl.paths import PathError, resolve_path
from eva_agent.graders_dsl.schema import validate_declarative_spec


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str


class DeclarativeGrader:
    def __init__(self, spec: GraderSpec) -> None:
        validate_declarative_spec(spec)
        self._spec = spec
        checks = spec.checks or []
        names = [check.effective_name for check in checks]
        if spec.score is not None and spec.score.weights:
            self._weights = {name: spec.score.weights.get(name, 1.0) for name in names}
        else:
            self._weights = {name: 1.0 for name in names}
        self._total_weight = sum(self._weights.values())

    def evaluate(self, record: dict[str, Any]) -> tuple[float, list[CheckResult]]:
        results: list[CheckResult] = []
        for check in self._spec.checks or []:
            value = _lookup(record, self._spec.input_map[check.field])
            passed = run_check(check, value)
            if passed:
                reason = ""
            elif value is MISSING:
                reason = f"field '{check.field}' missing"
            else:
                reason = f"{check.effective_name} failed"
            results.append(CheckResult(name=check.effective_name, passed=passed, reason=reason))

        earned = sum(self._weights[result.name] for result in results if result.passed)
        score = earned / self._total_weight if self._total_weight else 0.0
        return score, results


def _lookup(record: dict[str, Any], path: str) -> Any:
    try:
        return resolve_path(record, path)
    except PathError:
        return MISSING


def compile_grader(spec: GraderSpec) -> DeclarativeGrader:
    return DeclarativeGrader(spec)
