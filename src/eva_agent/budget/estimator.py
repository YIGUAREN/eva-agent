"""Local token counting and fail-closed model pricing."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from eva_agent.adapters.target.tokens import TokenEstimationError, estimate_tokens
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.grader import GraderSpec
from eva_agent.pricing import ModelPrice, PricingError, PricingTable


class BudgetPricingError(RuntimeError):
    """A call cannot be priced, so budget enforcement must refuse it."""


class CostEstimator:
    def __init__(self, pricing: PricingTable | None, *, token_encoding: str = "cl100k_base") -> None:
        self._pricing = pricing
        self._token_encoding = token_encoding

    def _price(self, model: ModelConfig) -> ModelPrice:
        if self._pricing is None:
            raise BudgetPricingError(
                f"缺少模型价格，请在 pricing.yaml 中补充 provider={model.provider!r} model={model.model!r}"
            )
        try:
            return self._pricing.get(model.provider, model.model)
        except PricingError as exc:
            raise BudgetPricingError(
                f"缺少模型价格，请在 pricing.yaml 中补充 provider={model.provider!r} model={model.model!r}"
            ) from exc

    def count_tokens(self, text: str) -> int:
        try:
            return estimate_tokens(text, self._token_encoding)
        except TokenEstimationError as exc:
            raise BudgetPricingError(f"无法估算 token：{exc}") from exc

    def cost_for_tokens(
        self,
        model: ModelConfig,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> Decimal:
        return self._price(model).cost_usd(input_tokens=input_tokens, output_tokens=output_tokens)

    def estimate_chat(
        self,
        model: ModelConfig,
        *,
        input_text: str,
        max_output_tokens: int,
    ) -> Decimal:
        return self.cost_for_tokens(
            model,
            input_tokens=self.count_tokens(input_text),
            output_tokens=max(0, max_output_tokens),
        )

    def estimate_grader(
        self,
        model: ModelConfig,
        *,
        spec: GraderSpec,
        rows: list[dict[str, Any]],
        max_output_tokens: int = 512,
    ) -> Decimal:
        payload = json.dumps(
            {"spec": spec.model_dump(mode="json"), "rows": rows},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return self.estimate_chat(model, input_text=payload, max_output_tokens=max_output_tokens)

    def estimate_aig_scan(
        self,
        model: ModelConfig,
        *,
        requests: int,
        max_output_tokens: int = 512,
    ) -> Decimal:
        count = max(1, requests)
        unit = self.cost_for_tokens(model, input_tokens=0, output_tokens=max(0, max_output_tokens))
        return unit * count

    def estimate_finalize(
        self,
        model: ModelConfig,
        *,
        input_text: str,
        max_output_tokens: int = 1024,
    ) -> Decimal:
        return self.estimate_chat(model, input_text=input_text, max_output_tokens=max_output_tokens)
