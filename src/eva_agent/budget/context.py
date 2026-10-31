"""Bind one run's ledger and pricing context to an async graph invocation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.config.settings import ModelConfig


@dataclass(frozen=True)
class BudgetModels:
    agent: ModelConfig
    judge: ModelConfig
    attack: ModelConfig


@dataclass(frozen=True)
class BudgetContext:
    ledger: BudgetLedger
    estimator: CostEstimator
    models: BudgetModels


_CURRENT: ContextVar[BudgetContext | None] = ContextVar("eva_budget_context", default=None)


@contextmanager
def bind_budget(
    ledger: BudgetLedger,
    estimator: CostEstimator,
    models: BudgetModels,
) -> Iterator[BudgetContext]:
    context = BudgetContext(ledger=ledger, estimator=estimator, models=models)
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


def budget_context() -> BudgetContext | None:
    return _CURRENT.get()
