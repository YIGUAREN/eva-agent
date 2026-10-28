"""Pricing lookup tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from eva_agent.pricing import ModelPrice, PricingTable
from eva_agent.tools.registry import ToolSpec


class GetPricingInput(BaseModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    run_id: str | None = None


def build_get_pricing_tool(table: PricingTable) -> ToolSpec:
    async def handler(payload: GetPricingInput) -> ModelPrice:
        return table.get(payload.provider, payload.model)

    return ToolSpec(
        name="get_pricing",
        description="Look up pricing for a provider/model; unknown models fail closed.",
        input_model=GetPricingInput,
        output_model=ModelPrice,
        permission="internal",
        timeout_s=5.0,
        side_effects=False,
        idempotent=True,
        handler=handler,
    )
