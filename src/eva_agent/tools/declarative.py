"""Run L2 declarative graders over probe records."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.graders_dsl.compiler import compile_grader
from eva_agent.tools.registry import ToolSpec


class RunDeclarativeChecksInput(BaseModel):
    specs: list[GraderSpec] = Field(min_length=1)
    records: list[dict[str, Any]]
    run_id: str | None = None


def build_run_declarative_checks_tool(*, max_rows: int) -> ToolSpec:
    async def handler(payload: RunDeclarativeChecksInput) -> GraderRunOutput:
        if len(payload.records) > max_rows:
            raise ValueError(f"records is {len(payload.records)}; max_rows is {max_rows}")
        started = time.perf_counter()
        outcomes: list[GraderSampleOutcome] = []
        for spec in payload.specs:
            if spec.level != "declarative":
                raise ValueError(f"grader '{spec.name}' is not declarative")
            grader = compile_grader(spec)
            for index, record in enumerate(payload.records):
                score, results = grader.evaluate(record)
                failed = [result.reason for result in results if not result.passed]
                outcomes.append(
                    GraderSampleOutcome(
                        grader=spec.name,
                        sample_index=index,
                        score=score,
                        raw_score=score,
                        reason="; ".join(failed),
                    )
                )
        return GraderRunOutput(
            outcomes=outcomes,
            duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )

    return ToolSpec(
        name="run_declarative_checks",
        description="Evaluate declarative grader specs against records.",
        input_model=RunDeclarativeChecksInput,
        output_model=GraderRunOutput,
        permission="internal",
        timeout_s=60.0,
        side_effects=False,
        idempotent=True,
        handler=handler,
    )
