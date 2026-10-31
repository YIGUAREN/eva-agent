"""Pricing table and cost calculation (fail closed on unknown models)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PricingError(ValueError):
    """Raised when pricing is missing or malformed."""


class UnknownModelPricingError(PricingError):
    """Raised when no price entry exists for a provider/model pair."""


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)

    def cost_usd(self, *, input_tokens: int, output_tokens: int) -> Decimal:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        return (
            Decimal(input_tokens) * self.input_per_million + Decimal(output_tokens) * self.output_per_million
        ) / Decimal(1_000_000)


class PricingTable:
    def __init__(self, entries: list[ModelPrice]) -> None:
        table: dict[tuple[str, str], ModelPrice] = {}
        for entry in entries:
            key = (entry.provider, entry.model)
            if key in table:
                raise PricingError(f"duplicate pricing entry: {entry.provider}/{entry.model}")
            table[key] = entry
        self._entries = table

    @classmethod
    def load(cls, path: Path) -> PricingTable:
        if not path.is_file():
            raise PricingError(f"pricing file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PricingError(f"invalid YAML in {path}: {exc}") from exc
        models = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models, list) or not models:
            raise PricingError(f"pricing file has no models: {path}")

        entries: list[ModelPrice] = []
        for index, item in enumerate(models):
            if not isinstance(item, dict):
                raise PricingError(f"pricing entry {index} must be a mapping")
            try:
                entries.append(ModelPrice.model_validate(item))
            except ValidationError as exc:
                details = "; ".join(
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in exc.errors(include_input=False, include_url=False)
                )
                raise PricingError(f"invalid pricing entry {index}: {details}") from None
        return cls(entries)

    def keys(self) -> list[tuple[str, str]]:
        return sorted(self._entries)

    def get(self, provider: str, model: str) -> ModelPrice:
        try:
            return self._entries[(provider, model)]
        except KeyError:
            raise UnknownModelPricingError(f"no pricing for provider={provider!r} model={model!r}") from None
