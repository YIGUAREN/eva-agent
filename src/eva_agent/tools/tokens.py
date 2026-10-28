"""Token counting tool backed by the configured tiktoken encoding."""

from __future__ import annotations

from pydantic import BaseModel, Field

from eva_agent.adapters.target.tokens import estimate_tokens
from eva_agent.tools.registry import ToolSpec


class CountTokensInput(BaseModel):
    text: str
    run_id: str | None = None


class CountTokensOutput(BaseModel):
    count: int = Field(ge=0)


def build_count_tokens_tool(encoding: str) -> ToolSpec:
    async def handler(payload: CountTokensInput) -> CountTokensOutput:
        return CountTokensOutput(count=estimate_tokens(payload.text, encoding))

    return ToolSpec(
        name="count_tokens",
        description="Count tokens with the configured tiktoken encoding.",
        input_model=CountTokensInput,
        output_model=CountTokensOutput,
        permission="internal",
        timeout_s=10.0,
        side_effects=False,
        idempotent=True,
        handler=handler,
    )
