"""Tools that expose the grader catalog and the OpenJudge runner."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from eva_agent.adapters.openjudge.catalog import GraderEntry, summarize
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderRunOutput
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


def build_run_graders_tool(runner: GraderRunner, *, max_rows: int) -> ToolSpec:
    async def handler(payload: RunGradersInput) -> GraderRunOutput:
        if len(payload.rows) > max_rows:
            raise ValueError(f"rows is {len(payload.rows)}; max_rows is {max_rows}")
        return await runner.run(payload.specs, payload.rows)

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
