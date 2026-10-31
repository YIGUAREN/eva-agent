"""Tools that expose the grader catalog and the OpenJudge runner."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from eva_agent.adapters.openjudge.cache import GradingCache, cache_key
from eva_agent.adapters.openjudge.catalog import GraderEntry, summarize
from eva_agent.budget.context import budget_context
from eva_agent.budget.estimator import BudgetPricingError
from eva_agent.budget.types import BudgetExhausted
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.tools.registry import ToolSpec


class GraderRunner(Protocol):
    async def run(self, specs: list[GraderSpec], rows: list[dict[str, Any]]) -> GraderRunOutput: ...


class ListGradersInput(BaseModel):
    tags: list[str] = Field(default_factory=list)
    run_id: str | None = None


class ListGradersOutput(BaseModel):
    graders: list[dict[str, Any]]


class RunGradersInput(BaseModel):
    specs: list[GraderSpec]
    rows: list[dict[str, Any]]
    run_id: str | None = None


def build_list_graders_tool(catalog: dict[str, GraderEntry]) -> ToolSpec:
    async def handler(payload: ListGradersInput) -> ListGradersOutput:
        entries = [
            entry
            for entry in catalog.values()
            if (set(payload.tags) <= set(entry.tags) if payload.tags else "skill" not in entry.tags)
        ]
        return ListGradersOutput(graders=summarize({entry.id: entry for entry in entries}))

    return ToolSpec(
        name="list_graders",
        description=(
            "List grader catalog entries with the inputs they require; "
            "package-level skill graders are listed only when a tag filter selects them."
        ),
        input_model=ListGradersInput,
        output_model=ListGradersOutput,
        permission="public_read",
        timeout_s=5.0,
        side_effects=False,
        idempotent=True,
        handler=handler,
    )


def build_run_graders_tool(
    runner: GraderRunner,
    *,
    max_rows: int,
    judge_model: ModelConfig | None = None,
    max_output_tokens: int = 512,
    data_dir: str | None = None,
) -> ToolSpec:
    async def handler(payload: RunGradersInput) -> GraderRunOutput:
        if len(payload.rows) > max_rows:
            raise ValueError(f"rows is {len(payload.rows)}; max_rows is {max_rows}")
        context = budget_context()
        enforced = context is not None and context.ledger.enforced
        if enforced and judge_model is None:
            # Fail closed: an unpriced judge call would run outside the budget.
            raise BudgetPricingError(
                "budget enforcement requires a judge model to price run_graders calls"
            )
        cache = (
            GradingCache(Path(data_dir), payload.run_id)
            if data_dir is not None and payload.run_id
            else None
        )
        combined = GraderRunOutput()
        for spec in payload.specs:
            cached_outcomes: list[GraderSampleOutcome] = []
            fresh: list[tuple[int, dict[str, Any], str]] = []
            for index, row in enumerate(payload.rows):
                key = cache_key(spec, row)
                hit = cache.read(spec.name, key, index) if cache is not None else None
                if hit is not None:
                    cached_outcomes.append(hit)
                else:
                    fresh.append((index, row, key))

            part: GraderRunOutput | None = None
            if fresh:
                fresh_rows = [row for _, row, _ in fresh]
                reserve = enforced
                estimate = Decimal("0")
                if reserve:
                    assert context is not None
                    assert judge_model is not None
                    estimate = context.estimator.estimate_grader(
                        judge_model,
                        spec=spec,
                        rows=fresh_rows,
                        max_output_tokens=max_output_tokens,
                    )
                    try:
                        await context.ledger.reserve(estimate, kind="grader")
                    except BudgetExhausted:
                        combined = _combine(combined, cached_outcomes)
                        return combined.model_copy(update={"truncated_reason": "budget"})
                try:
                    part = await runner.run([spec], fresh_rows)
                finally:
                    # Settle on every exit path, including cancellation: a
                    # ``wait_for`` timeout raises ``CancelledError``, which is a
                    # ``BaseException`` and would leak the reservation otherwise.
                    if reserve:
                        assert context is not None
                        assert judge_model is not None
                        actual = None
                        if part is not None and (part.input_tokens or part.output_tokens):
                            actual = context.estimator.cost_for_tokens(
                                judge_model,
                                input_tokens=part.input_tokens or 0,
                                output_tokens=part.output_tokens or 0,
                            )
                        await context.ledger.settle(estimate, actual, kind="grader")
                assert part is not None
                remapped: list[GraderSampleOutcome] = []
                for outcome in part.outcomes:
                    if outcome.sample_index >= len(fresh):
                        continue
                    original_index, _, key = fresh[outcome.sample_index]
                    adjusted = outcome.model_copy(update={"sample_index": original_index})
                    remapped.append(adjusted)
                    if cache is not None and outcome.error is None:
                        cache.write(spec.name, key, adjusted)
                part = part.model_copy(update={"outcomes": remapped, "fresh": len(remapped)})

            combined = _combine(combined, cached_outcomes, part)
        return combined

    return ToolSpec(
        name="run_graders",
        description="Run OpenJudge graders on prepared rows and return normalized outcomes.",
        input_model=RunGradersInput,
        output_model=GraderRunOutput,
        permission="internal",
        timeout_s=600.0,
        side_effects=False,
        idempotent=False,
        handler=handler,
    )


def _combine(
    combined: GraderRunOutput,
    cached: list[GraderSampleOutcome],
    part: GraderRunOutput | None = None,
) -> GraderRunOutput:
    part = part or GraderRunOutput()
    outcomes = sorted(
        [*combined.outcomes, *cached, *part.outcomes],
        key=lambda item: (item.grader, item.sample_index),
    )
    return GraderRunOutput(
        outcomes=outcomes,
        duration_ms=combined.duration_ms + part.duration_ms,
        error=combined.error or part.error,
        input_tokens=_add_optional(combined.input_tokens, part.input_tokens),
        output_tokens=_add_optional(combined.output_tokens, part.output_tokens),
        truncated_reason=combined.truncated_reason or part.truncated_reason,
        cached=combined.cached + len(cached) + part.cached,
        fresh=combined.fresh + part.fresh,
    )


def _add_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right
