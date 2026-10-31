import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.budget.types import RunUsage, quantize_money
from eva_agent.budget.usage import record_usage
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.pricing import ModelPrice, PricingTable
from eva_agent.tools.rubric import GenerateRubricInput, build_generate_rubric_tool


class _NullStore:
    async def load(self, run_id: str) -> RunUsage | None:
        return None

    async def save(self, run_id: str, usage: RunUsage) -> None:
        return None


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, task_description, sample_queries, scenario=None) -> list[str]:
        self.calls += 1
        return ["- rubric one"]


class UsageGenerator:
    async def generate(self, *, task_description, sample_queries, scenario=None) -> list[str]:
        record_usage(SimpleNamespace(usage={"prompt_tokens": 100, "completion_tokens": 50}))
        return ["- rubric one"]


class HangingGenerator:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def generate(self, *, task_description, sample_queries, scenario=None) -> list[str]:
        self.started.set()
        await asyncio.sleep(3600)
        raise AssertionError("cancelled before returning")


class ExplodingOnActualEstimator(CostEstimator):
    """Prices the estimate but fails when pricing the provider-reported actual."""

    def __init__(self, pricing_table: PricingTable) -> None:
        super().__init__(pricing_table)
        self.calls = 0

    def cost_for_tokens(
        self,
        model: ModelConfig,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("pricing failed")
        return super().cost_for_tokens(model, input_tokens=input_tokens, output_tokens=output_tokens)


def model() -> ModelConfig:
    return ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")


def price_table() -> PricingTable:
    return PricingTable(
        [
            ModelPrice(
                provider="anthropic",
                model="demo-echo",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            )
        ]
    )


async def test_rubric_generation_reserves_and_settles() -> None:
    generator = FakeGenerator()
    spec = build_generate_rubric_tool(generator, max_output_tokens=64)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    estimator = CostEstimator(
        PricingTable(
            [
                ModelPrice(
                    provider="anthropic",
                    model="demo-echo",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                )
            ]
        )
    )
    bound = BudgetModels(agent=model(), judge=model(), attack=model())
    with bind_budget(ledger, estimator, bound):
        output = await spec.handler(GenerateRubricInput(task_description="rate answers"))
    assert output.rubrics == ["- rubric one"]
    assert generator.calls == 1
    assert ledger.snapshot().spent_usd > 0
    assert ledger.estimated is True


async def test_rubric_settles_the_reservation_when_the_task_is_cancelled() -> None:
    generator = HangingGenerator()
    spec = build_generate_rubric_tool(generator, max_output_tokens=64)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    bound = BudgetModels(agent=model(), judge=model(), attack=model())

    async def run():
        return await spec.handler(GenerateRubricInput(task_description="rate answers"))

    with bind_budget(ledger, CostEstimator(price_table()), bound):
        task = asyncio.create_task(run())
        await generator.started.wait()
        assert ledger.snapshot().reserved_usd > 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.snapshot().spent_usd > 0
    assert ledger.estimated is True


async def test_rubric_settles_the_reservation_when_wait_for_times_out() -> None:
    generator = HangingGenerator()
    spec = build_generate_rubric_tool(generator, max_output_tokens=64)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    bound = BudgetModels(agent=model(), judge=model(), attack=model())

    async def run():
        return await spec.handler(GenerateRubricInput(task_description="rate answers"))

    with bind_budget(ledger, CostEstimator(price_table()), bound):
        task = asyncio.create_task(run())
        await generator.started.wait()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(task, timeout=0.01)
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.snapshot().spent_usd > 0
    assert ledger.estimated is True


async def test_rubric_settles_the_estimate_when_pricing_the_actual_cost_fails() -> None:
    generator = UsageGenerator()
    spec = build_generate_rubric_tool(generator, max_output_tokens=64)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    expected = quantize_money(
        CostEstimator(price_table()).estimate_chat(
            model(), input_text="rate answers", max_output_tokens=64
        )
    )
    bound = BudgetModels(agent=model(), judge=model(), attack=model())
    with bind_budget(ledger, ExplodingOnActualEstimator(price_table()), bound):
        await spec.handler(GenerateRubricInput(task_description="rate answers"))
    assert ledger.snapshot().spent_usd == expected
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.estimated is True
