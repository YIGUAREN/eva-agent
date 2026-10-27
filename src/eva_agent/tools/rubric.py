"""L1 rubric generation tool."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

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


def build_generate_rubric_tool(generator: RubricGenerator) -> ToolSpec:
    async def handler(payload: GenerateRubricInput) -> RubricOutput:
        rubrics = await generator.generate(
            task_description=payload.task_description,
            sample_queries=payload.sample_queries,
            scenario=payload.scenario,
        )
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
