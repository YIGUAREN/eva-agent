from eva_agent.budget.context import BudgetModels, bind_budget, budget_context
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.budget.types import RunUsage
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget


class _NullStore:
    async def load(self, run_id: str) -> RunUsage | None:
        return None

    async def save(self, run_id: str, usage: RunUsage) -> None:
        return None


def models() -> BudgetModels:
    model = ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")
    return BudgetModels(agent=model, judge=model, attack=model)


def test_bind_budget_exposes_then_restores_the_context() -> None:
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=1.0, max_target_requests=1),
        store=_NullStore(),
    )
    assert budget_context() is None
    with bind_budget(ledger, CostEstimator(None), models()) as bound:
        assert bound is not None
        assert budget_context() is not None
        context = budget_context()
        assert context is not None
        assert context.ledger is ledger
    assert budget_context() is None
