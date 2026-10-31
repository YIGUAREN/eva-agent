"""Estimate the money the merge/analyze/report tail needs before it starts."""

from __future__ import annotations

import json
from decimal import Decimal

from eva_agent.budget.estimator import CostEstimator
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.probe import ProbeSummary
from eva_agent.domain.result import MergedResult

FINALIZE_MAX_OUTPUT_TOKENS = 1024


def finalize_input(
    *,
    plan: EvalPlan | None,
    merged: MergedResult | None,
    probe: ProbeSummary | None,
) -> str:
    payload = {
        "plan": plan.model_dump(mode="json") if plan is not None else None,
        "dimensions": [item.model_dump(mode="json") for item in merged.results] if merged is not None else [],
        "skipped": [item.model_dump(mode="json") for item in merged.skipped] if merged is not None else [],
        "probe": probe.model_dump(mode="json") if probe is not None else None,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def estimate_finalize(
    estimator: CostEstimator,
    model: ModelConfig,
    *,
    plan: EvalPlan | None,
    merged: MergedResult | None = None,
    probe: ProbeSummary | None = None,
    max_output_tokens: int = FINALIZE_MAX_OUTPUT_TOKENS,
) -> Decimal:
    if plan is None:
        return Decimal("0")
    return estimator.estimate_chat(
        model,
        input_text=finalize_input(plan=plan, merged=merged, probe=probe),
        max_output_tokens=max_output_tokens,
    )
