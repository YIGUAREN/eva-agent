from decimal import Decimal

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.budget.types import RunUsage
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


def model() -> ModelConfig:
    return ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")


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
