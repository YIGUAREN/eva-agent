"""L1 rubric generation tool."""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, Field

from eva_agent.budget.context import budget_context
from eva_agent.budget.types import BudgetExhausted
from eva_agent.budget.usage import track_usage
from eva_agent.tools.registry import ToolSpec


class RubricGenerator(Protocol):
    async def generate(
        self,
        *,
        task_description: str,
        sample_queries: list[str],
        scenario: str | None = None,
    ) -> list[str]: ...


class GenerateRubricInput(BaseModel):
    task_description: str = Field(min_length=1)
    sample_queries: list[str] = Field(default_factory=list)
    scenario: str | None = None
    run_id: str | None = None


class RubricOutput(BaseModel):
    rubrics: list[str]


def build_generate_rubric_tool(
    generator: RubricGenerator,
    *,
    max_output_tokens: int = 1024,
) -> ToolSpec:
    async def handler(payload: GenerateRubricInput) -> RubricOutput:
        context = budget_context()
        enforced = context is not None and context.ledger.enforced
        estimate = Decimal("0")
        reserve_attempted = False
        reserved = False
        try:
            if enforced:
                assert context is not None
                input_text = "\n".join([payload.task_description, *payload.sample_queries])
                estimate = context.estimator.estimate_chat(
                    context.models.judge,
                    input_text=input_text,
                    max_output_tokens=max_output_tokens,
                )
                reserve_attempted = True
                await context.ledger.reserve(estimate, kind="rubric")
                reserved = True
            with track_usage() as totals:
                rubrics = await generator.generate(
                    task_description=payload.task_description,
                    sample_queries=payload.sample_queries,
                    scenario=payload.scenario,
                )
        except BudgetExhausted:
            # `reserve` refusing charges nothing, but the guarded call can also
            # raise this after a successful reservation: settle it then too.
            if enforced and reserved:
                assert context is not None
                await context.ledger.settle(estimate, None, kind="rubric")
            raise
        except BaseException:
            # The registry enforces timeout_s with asyncio.wait_for, which
            # cancels the handler; CancelledError is a BaseException and must
            # still release the reservation.
            if enforced and (reserved or reserve_attempted):
                assert context is not None
                await context.ledger.settle(estimate, None, kind="rubric")
            raise
        if enforced:
            assert context is not None
            actual = None
            try:
                # Match the planner: one-sided or zero usage cannot be priced
                # reliably, so settle the estimate and mark the ledger estimated.
                if totals.input_tokens > 0 and totals.output_tokens > 0:
                    actual = context.estimator.cost_for_tokens(
                        context.models.judge,
                        input_tokens=totals.input_tokens,
                        output_tokens=totals.output_tokens,
                    )
            except Exception:
                # Pricing the actual cost must never orphan the reservation.
                actual = None
            await context.ledger.settle(estimate, actual, kind="rubric")
        return RubricOutput(rubrics=rubrics)

    return ToolSpec(
        name="generate_rubric",
        description="Generate L1 evaluation rubrics with the judge model before approval.",
        input_model=GenerateRubricInput,
        output_model=RubricOutput,
        permission="internal",
        timeout_s=120.0,
        side_effects=False,
        idempotent=True,
        handler=handler,
    )
