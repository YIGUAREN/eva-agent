"""Money-safe budget values shared by the ledger, storage, and API views."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol

MONEY_QUANT = Decimal("0.000001")


def quantize_money(value: Decimal) -> Decimal:
    """Round a Decimal to the six decimal places used by every money column."""
    return value.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class RunUsage:
    spent_usd: Decimal = Decimal("0")
    reserved_usd: Decimal = Decimal("0")
    spent_wall_clock_s: float = 0.0
    target_requests: int = 0
    aig_requests: int = 0


@dataclass(frozen=True)
class BudgetSnapshot:
    spent_usd: Decimal
    reserved_usd: Decimal
    remaining_usd: Decimal
    spent_wall_clock_s: float
    target_requests: int
    aig_requests: int
    estimated: bool


class BudgetExhausted(RuntimeError):
    """The next call would exceed a budget limit."""

    def __init__(self, *, spent: Decimal, remaining: Decimal, reason: str) -> None:
        super().__init__(f"budget exhausted: {reason}")
        self.spent = spent
        self.remaining = remaining
        self.reason = reason


class LedgerStore(Protocol):
    async def load(self, run_id: str) -> RunUsage | None: ...

    async def save(self, run_id: str, usage: RunUsage) -> None: ...
