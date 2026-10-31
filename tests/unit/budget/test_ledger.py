from decimal import Decimal

import pytest

from eva_agent.budget.ledger import BudgetLedger
from eva_agent.budget.types import BudgetExhausted, RunUsage
from eva_agent.domain.budget import Budget


class FakeStore:
    def __init__(self) -> None:
        self.saved: list[RunUsage] = []

    async def load(self, run_id: str) -> RunUsage | None:
        return None

    async def save(self, run_id: str, usage: RunUsage) -> None:
        self.saved.append(usage)


def money_budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=2)


def test_reserve_settle_and_snapshot() -> None:
    ledger = BudgetLedger(money_budget())
    ledger._reserve_sync(Decimal("0.5"))
    ledger._settle_sync(Decimal("0.5"), Decimal("0.2"))
    snapshot = ledger.snapshot()
    assert snapshot.spent_usd == Decimal("0.2")
    assert snapshot.reserved_usd == 0
    assert snapshot.remaining_usd == Decimal("0.8")


def test_reserve_rejects_an_estimate_over_the_remaining_money() -> None:
    ledger = BudgetLedger(money_budget())
    ledger._reserve_sync(Decimal("0.6"))
    with pytest.raises(BudgetExhausted) as exc:
        ledger._reserve_sync(Decimal("0.5"))
    assert exc.value.spent == Decimal("0")
    assert exc.value.remaining == Decimal("0.4")


def test_unknown_actual_settles_at_the_estimate_and_marks_estimated() -> None:
    ledger = BudgetLedger(money_budget())
    ledger._reserve_sync(Decimal("0.4"))
    ledger._settle_sync(Decimal("0.4"), None)
    assert ledger.snapshot().spent_usd == Decimal("0.4")
    assert ledger.estimated is True


def test_request_and_scan_limits() -> None:
    ledger = BudgetLedger(money_budget())
    ledger._reserve_requests_sync(2)
    with pytest.raises(BudgetExhausted):
        ledger._reserve_requests_sync(1)
    with pytest.raises(BudgetExhausted):
        ledger._reserve_scan_sync(1, Decimal("0.1"))
    assert ledger.snapshot().target_requests == 2


def test_unlimited_ledger_never_raises_but_tracks() -> None:
    ledger = BudgetLedger(money_budget(), unlimited=True)
    ledger._reserve_sync(Decimal("99"))
    ledger._settle_sync(Decimal("99"), Decimal("99"))
    assert ledger.enforced is False
    assert ledger.snapshot().spent_usd == Decimal("99")


async def test_persistence_is_throttled_and_flushed() -> None:
    now = [0.0]
    store = FakeStore()
    ledger = BudgetLedger(money_budget(), store=store, run_id="run-1", clock=lambda: now[0])
    await ledger.reserve(Decimal("0.1"), kind="test")
    assert store.saved == []
    now[0] = 1.0
    await ledger.reserve(Decimal("0.1"), kind="test")
    assert len(store.saved) == 1
    now[0] = 1.2
    await ledger.settle(Decimal("0.1"), Decimal("0.05"), kind="test")
    assert len(store.saved) == 1
    await ledger.flush()
    assert len(store.saved) == 2
    assert store.saved[-1].spent_usd == Decimal("0.05")


async def test_update_budget_changes_the_limit_and_remaining() -> None:
    ledger = BudgetLedger(money_budget())
    await ledger.reserve(Decimal("0.9"), kind="test")
    ledger.update_budget(Budget(max_cost_usd=2.0, max_wall_clock_s=120.0, max_target_requests=5))
    assert ledger.snapshot().remaining_usd == Decimal("1.1")
