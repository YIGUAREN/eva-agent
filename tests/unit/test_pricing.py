from decimal import Decimal
from pathlib import Path

import pytest

from eva_agent.pricing import ModelPrice, PricingError, PricingTable


def write_pricing(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pricing.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_and_compute_cost(tmp_path: Path):
    path = write_pricing(
        tmp_path,
        """
models:
  - provider: demo
    model: demo-echo
    input_per_million: 0.27
    output_per_million: 1.10
""",
    )

    price = PricingTable.load(path).get("demo", "demo-echo")

    assert price.cost_usd(input_tokens=1000, output_tokens=500) == Decimal("0.00082")


def test_unknown_model_fails_closed(tmp_path: Path):
    path = write_pricing(
        tmp_path,
        """
models:
  - provider: demo
    model: demo-echo
    input_per_million: 0.27
    output_per_million: 1.10
""",
    )

    with pytest.raises(PricingError, match="no pricing"):
        PricingTable.load(path).get("demo", "unknown")


def test_duplicate_entry_rejected():
    price = ModelPrice(provider="demo", model="m", input_per_million=Decimal("1"), output_per_million=Decimal("1"))

    with pytest.raises(PricingError, match="duplicate pricing"):
        PricingTable([price, price])


def test_negative_tokens_rejected():
    price = ModelPrice(provider="demo", model="m", input_per_million=Decimal("1"), output_per_million=Decimal("1"))

    with pytest.raises(ValueError):
        price.cost_usd(input_tokens=-1, output_tokens=0)


def test_missing_file_rejected(tmp_path: Path):
    with pytest.raises(PricingError, match="not found"):
        PricingTable.load(tmp_path / "missing.yaml")
