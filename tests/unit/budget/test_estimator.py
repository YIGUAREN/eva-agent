from decimal import Decimal

import pytest

from eva_agent.adapters.target import tokens as tokens_module
from eva_agent.budget.estimator import BudgetPricingError, CostEstimator
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.grader import GraderSpec
from eva_agent.pricing import ModelPrice, PricingTable

PROVIDER = "anthropic"


class FakeEncoding:
    """Offline stand-in for ``tiktoken.Encoding``: one token per character."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    tokens_module._encoding.cache_clear()
    monkeypatch.setattr(tokens_module.tiktoken, "get_encoding", lambda name: FakeEncoding())
    yield
    tokens_module._encoding.cache_clear()


def model() -> ModelConfig:
    return ModelConfig(provider=PROVIDER, model="demo-echo", api_key_env="DEMO_KEY")


def table() -> PricingTable:
    return PricingTable(
        [
            ModelPrice(
                provider=PROVIDER,
                model="demo-echo",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            )
        ]
    )


def test_unknown_model_fails_closed() -> None:
    estimator = CostEstimator(table())
    other = ModelConfig(provider=PROVIDER, model="missing", api_key_env="DEMO_KEY")
    with pytest.raises(BudgetPricingError, match="pricing.yaml"):
        estimator.estimate_chat(other, input_text="hello", max_output_tokens=10)


def test_missing_pricing_table_fails_closed() -> None:
    with pytest.raises(BudgetPricingError):
        CostEstimator(None).estimate_chat(model(), input_text="x", max_output_tokens=10)


def test_chat_and_grader_estimates_are_positive_decimals() -> None:
    estimator = CostEstimator(table())
    chat = estimator.estimate_chat(model(), input_text="hello world", max_output_tokens=100)
    grader = estimator.estimate_grader(
        model(),
        spec=GraderSpec(level="llm_rubric", name="quality", rubric="- clear"),
        rows=[{"query": "q", "response": "a"}],
    )
    assert isinstance(chat, Decimal) and chat > 0
    assert isinstance(grader, Decimal) and grader > 0


def test_aig_estimate_scales_with_the_request_count() -> None:
    estimator = CostEstimator(table())
    one = estimator.estimate_aig_scan(model(), requests=1)
    three = estimator.estimate_aig_scan(model(), requests=3)
    assert three == one * 3


def test_cost_for_tokens_uses_the_model_price() -> None:
    estimator = CostEstimator(table())
    assert estimator.cost_for_tokens(model(), input_tokens=1_000_000, output_tokens=0) == Decimal("1.000000")
    assert estimator.cost_for_tokens(model(), input_tokens=0, output_tokens=1_000_000) == Decimal("2.000000")
