from decimal import Decimal
from types import SimpleNamespace

import pytest

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.budget.types import BudgetExhausted, RunUsage
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.graph.planner import LlmPlanner
from eva_agent.pricing import ModelPrice, PricingTable
from eva_agent.tools.registry import ToolRegistry


class _NullStore:
    async def load(self, run_id: str) -> RunUsage | None:
        return None

    async def save(self, run_id: str, usage: RunUsage) -> None:
        return None


class FakeStructured:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response or SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 50})
        self.error = error
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response


def model() -> ModelConfig:
    return ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")


def pricing() -> CostEstimator:
    return CostEstimator(
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


def models() -> BudgetModels:
    return BudgetModels(agent=model(), judge=model(), attack=model())


async def test_planner_reserves_then_settles_actual_tokens() -> None:
    structured = FakeStructured()
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    with bind_budget(ledger, pricing(), models()):
        await planner._invoke_structured("hello")
    assert len(structured.prompts) == 1
    assert ledger.snapshot().spent_usd == Decimal("0.000200")
    assert ledger.snapshot().reserved_usd == 0


async def test_planner_raises_before_the_call_when_the_budget_is_exhausted() -> None:
    structured = FakeStructured()
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=1_000_000)
    ledger = BudgetLedger(
        Budget(max_cost_usd=0.000001, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    with bind_budget(ledger, pricing(), models()), pytest.raises(BudgetExhausted):
        await planner._invoke_structured("hello")
    assert structured.prompts == []


async def test_planner_settles_the_estimate_when_the_call_fails() -> None:
    structured = FakeStructured(error=RuntimeError("boom"))
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=100)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    with bind_budget(ledger, pricing(), models()), pytest.raises(RuntimeError):
        await planner._invoke_structured("hello")
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.snapshot().spent_usd > 0
    assert ledger.estimated is True
