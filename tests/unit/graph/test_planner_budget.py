import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.budget.types import BudgetExhausted, RunUsage, quantize_money
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


class HangingStructured:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def ainvoke(self, prompt: str):
        self.started.set()
        await asyncio.sleep(3600)


class ExhaustedStructured:
    """Raises `BudgetExhausted` from the guarded call, after a successful reserve."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        raise BudgetExhausted(spent=Decimal("0"), remaining=Decimal("0"), reason="max_cost_usd")


class FailOnceStore:
    """A store whose first save raises, simulating a persistence failure after reserve."""

    def __init__(self) -> None:
        self.saves = 0

    async def load(self, run_id: str) -> RunUsage | None:
        return None

    async def save(self, run_id: str, usage: RunUsage) -> None:
        self.saves += 1
        if self.saves == 1:
            raise RuntimeError("store down")


class RawStructured:
    def __init__(
        self,
        *,
        raw: object | None = None,
        parsed: object | None = None,
        parsing_error: BaseException | None = None,
    ) -> None:
        self.raw = (
            raw
            if raw is not None
            else SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 50})
        )
        self.parsed = parsed if parsed is not None else SimpleNamespace()
        self.parsing_error = parsing_error
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str):
        self.prompts.append(prompt)
        return {"raw": self.raw, "parsed": self.parsed, "parsing_error": self.parsing_error}


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


def pricing() -> CostEstimator:
    return CostEstimator(price_table())


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


async def test_planner_settles_the_reservation_when_the_task_is_cancelled() -> None:
    structured = HangingStructured()
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    with bind_budget(ledger, pricing(), models()):
        task = asyncio.create_task(planner._invoke_structured("hello"))
        await asyncio.wait_for(structured.started.wait(), timeout=1)
        assert ledger.snapshot().reserved_usd > 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.snapshot().spent_usd > 0
    assert ledger.estimated is True


async def test_planner_settles_the_reservation_when_wait_for_times_out() -> None:
    structured = HangingStructured()
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    with bind_budget(ledger, pricing(), models()):
        task = asyncio.create_task(planner._invoke_structured("hello"))
        await asyncio.wait_for(structured.started.wait(), timeout=1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(task, timeout=0.01)
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.snapshot().spent_usd > 0
    assert ledger.estimated is True


async def test_planner_prices_the_raw_message_when_structured_output_includes_it() -> None:
    parsed = SimpleNamespace()
    structured = RawStructured(parsed=parsed)
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    with bind_budget(ledger, pricing(), models()):
        output = await planner._invoke_structured("hello")
    assert output is parsed
    assert ledger.snapshot().spent_usd == Decimal("0.000200")
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.estimated is False


async def test_planner_settles_the_estimate_when_structured_parsing_fails() -> None:
    structured = RawStructured(parsing_error=ValueError("bad json"))
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    with bind_budget(ledger, pricing(), models()), pytest.raises(ValueError):
        await planner._invoke_structured("hello")
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.snapshot().spent_usd > 0
    assert ledger.estimated is True


async def test_planner_settles_the_estimate_when_the_response_has_no_usage() -> None:
    structured = FakeStructured(response=SimpleNamespace())
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    expected = quantize_money(pricing().estimate_chat(model(), input_text="hello", max_output_tokens=10))
    with bind_budget(ledger, pricing(), models()):
        await planner._invoke_structured("hello")
    assert ledger.snapshot().spent_usd == expected
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.estimated is True


@pytest.mark.parametrize(
    "metadata",
    [
        {"input_tokens": 0, "output_tokens": 0},
        {"input_tokens": 100},
        {"output_tokens": 50},
    ],
)
async def test_planner_treats_partial_or_zero_usage_as_unavailable(metadata: dict[str, int]) -> None:
    structured = FakeStructured(response=SimpleNamespace(usage_metadata=metadata))
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    expected = quantize_money(pricing().estimate_chat(model(), input_text="hello", max_output_tokens=10))
    with bind_budget(ledger, pricing(), models()):
        await planner._invoke_structured("hello")
    assert ledger.snapshot().spent_usd == expected
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.estimated is True


async def test_planner_settles_the_estimate_when_pricing_the_actual_cost_fails() -> None:
    structured = FakeStructured()
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    expected = quantize_money(pricing().estimate_chat(model(), input_text="hello", max_output_tokens=10))
    estimator = ExplodingOnActualEstimator(price_table())
    with bind_budget(ledger, estimator, models()):
        await planner._invoke_structured("hello")
    assert ledger.snapshot().spent_usd == expected
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.estimated is True


async def test_planner_settles_the_reservation_when_the_call_raises_budget_exhausted() -> None:
    structured = ExhaustedStructured()
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=_NullStore(),
    )
    expected = quantize_money(pricing().estimate_chat(model(), input_text="hello", max_output_tokens=10))
    with bind_budget(ledger, pricing(), models()), pytest.raises(BudgetExhausted):
        await planner._invoke_structured("hello")
    assert structured.prompts == ["hello"]
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.snapshot().spent_usd == expected
    assert ledger.estimated is True


async def test_planner_settles_when_the_store_fails_after_reserve() -> None:
    structured = FakeStructured()
    planner = LlmPlanner(None, ToolRegistry(), structured_runner=structured, max_output_tokens=10)
    store = FailOnceStore()
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5),
        store=store,
        run_id="planner-store-failure",
        throttle_s=0,
    )
    expected = quantize_money(pricing().estimate_chat(model(), input_text="hello", max_output_tokens=10))
    with bind_budget(ledger, pricing(), models()), pytest.raises(RuntimeError, match="store down"):
        await planner._invoke_structured("hello")
    assert structured.prompts == []
    assert ledger.snapshot().reserved_usd == 0
    assert ledger.snapshot().spent_usd == expected
    assert ledger.estimated is True
