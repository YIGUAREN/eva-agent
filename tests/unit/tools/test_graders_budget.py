from decimal import Decimal
from typing import Any

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.pricing import ModelPrice, PricingTable
from eva_agent.tools.graders import RunGradersInput, build_run_graders_tool


class CountingRunner:
    def __init__(self) -> None:
        self.calls = 0
        self.spec_names: list[str] = []

    async def run(self, specs: list[GraderSpec], rows: list[dict[str, Any]]) -> GraderRunOutput:
        self.calls += 1
        self.spec_names.extend(spec.name for spec in specs)
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(grader=spec.name, sample_index=index, score=0.8)
                for spec in specs
                for index in range(len(rows))
            ],
            duration_ms=1,
            input_tokens=10,
            output_tokens=1000,
        )


class NoopStore:
    async def load(self, run_id: str) -> None:
        return None

    async def save(self, run_id: str, usage: Any) -> None:
        return None


def model() -> ModelConfig:
    return ModelConfig(
        provider="litellm",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
        api_key_env="DEMO_KEY",
    )


def estimator() -> CostEstimator:
    return CostEstimator(
        PricingTable(
            [
                ModelPrice(
                    provider="litellm",
                    model="demo-echo",
                    input_per_million=Decimal("0"),
                    output_per_million=Decimal("1"),
                )
            ]
        )
    )


async def test_graders_stop_after_the_first_spec_that_exceeds_the_budget() -> None:
    runner = CountingRunner()
    spec = build_run_graders_tool(runner, max_rows=10, judge_model=model(), max_output_tokens=512)
    ledger = BudgetLedger(
        Budget(max_cost_usd=0.0012, max_wall_clock_s=60.0, max_target_requests=10),
        store=NoopStore(),
    )
    bound = BudgetModels(agent=model(), judge=model(), attack=model())
    specs = [
        GraderSpec(level="builtin", name="relevance", builtin_id="relevance"),
        GraderSpec(level="llm_rubric", name="quality", rubric="- clear"),
    ]
    with bind_budget(ledger, estimator(), bound):
        assert spec.handler is not None
        output = await spec.handler(
            RunGradersInput(specs=specs, rows=[{"query": "q", "response": "a"}], run_id="run-1")
        )
    assert runner.calls == 1
    assert runner.spec_names == ["relevance"]
    assert output.truncated_reason == "budget"
    assert {item.grader for item in output.outcomes} == {"relevance"}


async def test_graders_without_a_budget_context_run_every_spec() -> None:
    runner = CountingRunner()
    spec = build_run_graders_tool(runner, max_rows=10, judge_model=model())
    specs = [
        GraderSpec(level="builtin", name="relevance", builtin_id="relevance"),
        GraderSpec(level="llm_rubric", name="quality", rubric="- clear"),
    ]
    assert spec.handler is not None
    output = await spec.handler(RunGradersInput(specs=specs, rows=[{"query": "q", "response": "a"}]))
    assert runner.calls == 2
    assert output.truncated_reason is None
