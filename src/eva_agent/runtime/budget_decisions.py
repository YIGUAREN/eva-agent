"""Validation shared by the manager and the HTTP route."""

from __future__ import annotations

from decimal import Decimal

from eva_agent.budget.types import RunUsage
from eva_agent.domain.budget import Budget


class BudgetOverrunRequired(RuntimeError):
    def __init__(self, *, finalize_estimate: Decimal, remaining: Decimal) -> None:
        super().__init__("budget overrun requires confirmation")
        self.finalize_estimate = finalize_estimate
        self.remaining = remaining


class InvalidBudgetExtension(ValueError):
    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field


def ensure_summarize_allowed(
    *,
    finalize_estimate: Decimal,
    remaining: Decimal,
    confirm_overrun: bool,
) -> None:
    if remaining < finalize_estimate and not confirm_overrun:
        raise BudgetOverrunRequired(finalize_estimate=finalize_estimate, remaining=remaining)


def validate_extension(budget: Budget, usage: RunUsage) -> None:
    # Messages stay free of spend amounts: they cross the public error envelope,
    # while exact figures are exposed separately through the details field.
    if Decimal(str(budget.max_cost_usd)) <= usage.spent_usd:
        raise InvalidBudgetExtension(
            "max_cost_usd",
            "新预算 max_cost_usd 必须严格大于已消费金额",
        )
    if float(budget.max_wall_clock_s) <= usage.spent_wall_clock_s:
        raise InvalidBudgetExtension(
            "max_wall_clock_s",
            "新预算 max_wall_clock_s 必须严格大于已消耗时间",
        )
    consumed_requests = max(usage.target_requests, usage.aig_requests)
    if budget.max_target_requests <= consumed_requests:
        raise InvalidBudgetExtension(
            "max_target_requests",
            "新预算 max_target_requests 必须严格大于已消耗请求数",
        )
