from decimal import Decimal

import pytest

from eva_agent.domain.grader import CheckSpec, GraderSpec
from eva_agent.pricing import ModelPrice, PricingTable
from eva_agent.tools.declarative import RunDeclarativeChecksInput, build_run_declarative_checks_tool
from eva_agent.tools.pricing import GetPricingInput, build_get_pricing_tool
from eva_agent.tools.tokens import CountTokensInput, build_count_tokens_tool


async def test_count_tokens_tool(monkeypatch: pytest.MonkeyPatch):
    from eva_agent.adapters.target import tokens

    class _FixedEncoder:
        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

    tokens._encoding.cache_clear()
    monkeypatch.setattr(tokens.tiktoken, "get_encoding", lambda name: _FixedEncoder())
    try:
        tool = build_count_tokens_tool("cl100k_base")
        output = await tool.handler(CountTokensInput(text="hello world"))
    finally:
        tokens._encoding.cache_clear()

    assert output.count == 11


async def test_get_pricing_tool():
    price = ModelPrice(
        provider="demo", model="demo-echo", input_per_million=Decimal(0), output_per_million=Decimal(0)
    )
    tool = build_get_pricing_tool(PricingTable([price]))

    output = await tool.handler(GetPricingInput(provider="demo", model="demo-echo"))

    assert output == price


def latency_spec() -> GraderSpec:
    return GraderSpec(
        level="declarative",
        name="latency_budget",
        input_map={"latency_ms": "$.latency_ms"},
        checks=[CheckSpec(type="numeric_threshold", field="latency_ms", op="<=", value=100)],
    )


async def test_declarative_tool_scores_each_record():
    tool = build_run_declarative_checks_tool(max_rows=10)

    output = await tool.handler(
        RunDeclarativeChecksInput(
            specs=[latency_spec()],
            records=[{"latency_ms": 50}, {"latency_ms": 150}],
        )
    )

    assert [outcome.score for outcome in output.outcomes] == [1.0, 0.0]
    assert output.outcomes[0].grader == "latency_budget"


async def test_declarative_tool_rejects_non_declarative_spec():
    tool = build_run_declarative_checks_tool(max_rows=10)

    with pytest.raises(ValueError, match="declarative"):
        await tool.handler(
            RunDeclarativeChecksInput(
                specs=[GraderSpec(level="builtin", name="x", builtin_id="relevance")],
                records=[{"latency_ms": 1}],
            )
        )


async def test_declarative_tool_enforces_row_limit():
    tool = build_run_declarative_checks_tool(max_rows=1)

    with pytest.raises(ValueError, match="max_rows"):
        await tool.handler(
            RunDeclarativeChecksInput(specs=[latency_spec()], records=[{"latency_ms": 1}, {"latency_ms": 2}])
        )
