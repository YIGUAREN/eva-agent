import asyncio
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


class SlowStore(FakeStore):
    """A store that exposes whether two saves ever overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def save(self, run_id: str, usage: RunUsage) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        await super().save(run_id, usage)
        self.active -= 1


def money_budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=2)


def test_reserve_settle_and_snapshot() -> None:
    ledger = BudgetLedger(money_budget(), store=FakeStore())
    ledger._reserve_sync(Decimal("0.5"))
    ledger._settle_sync(Decimal("0.5"), Decimal("0.2"))
    snapshot = ledger.snapshot()
    assert snapshot.spent_usd == Decimal("0.2")
    assert snapshot.reserved_usd == 0
    assert snapshot.remaining_usd == Decimal("0.8")


def test_reserve_rejects_an_estimate_over_the_remaining_money() -> None:
    ledger = BudgetLedger(money_budget(), store=FakeStore())
    ledger._reserve_sync(Decimal("0.6"))
    with pytest.raises(BudgetExhausted) as exc:
        ledger._reserve_sync(Decimal("0.5"))
    assert exc.value.spent == Decimal("0")
    assert exc.value.remaining == Decimal("0.4")


def test_unknown_actual_settles_at_the_estimate_and_marks_estimated() -> None:
    ledger = BudgetLedger(money_budget(), store=FakeStore())
    ledger._reserve_sync(Decimal("0.4"))
    ledger._settle_sync(Decimal("0.4"), None)
    assert ledger.snapshot().spent_usd == Decimal("0.4")
    assert ledger.estimated is True


def test_request_and_scan_limits() -> None:
    ledger = BudgetLedger(money_budget(), store=FakeStore())
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


def test_limited_ledger_requires_a_store() -> None:
    with pytest.raises(ValueError):
        BudgetLedger(money_budget())


def test_enforced_follows_the_unlimited_flag() -> None:
    assert BudgetLedger(money_budget(), store=FakeStore()).enforced is True
    assert BudgetLedger.unlimited(money_budget()).enforced is False
    assert BudgetLedger(money_budget(), unlimited=True).enforced is False


def test_wall_clock_elapsed_is_not_double_counted_across_mutations() -> None:
    now = [0.0]
    ledger = BudgetLedger(money_budget(), store=FakeStore(), clock=lambda: now[0])
    now[0] = 1.0
    ledger._reserve_sync(Decimal("0.1"))
    now[0] = 2.0
    ledger._settle_sync(Decimal("0.1"), Decimal("0.1"))
    now[0] = 3.0
    ledger._reserve_requests_sync(1)
    assert ledger.snapshot().spent_wall_clock_s == pytest.approx(3.0)


def test_wall_clock_budget_does_not_fire_early_but_fires_when_exhausted() -> None:
    now = [0.0]
    budget = Budget(max_cost_usd=1.0, max_wall_clock_s=5.0, max_target_requests=5)
    ledger = BudgetLedger(budget, store=FakeStore(), clock=lambda: now[0])
    for elapsed in (1.0, 2.0, 3.0):
        now[0] = elapsed
        ledger._reserve_requests_sync(1)
    assert ledger.snapshot().spent_wall_clock_s == pytest.approx(3.0)
    now[0] = 5.5
    with pytest.raises(BudgetExhausted) as exc:
        ledger._reserve_requests_sync(1)
    assert exc.value.reason == "max_wall_clock_s"


def test_scan_rejects_a_negative_estimate_without_freeing_headroom() -> None:
    ledger = BudgetLedger(money_budget(), store=FakeStore())
    ledger._reserve_sync(Decimal("0.6"))
    with pytest.raises(ValueError):
        ledger._reserve_scan_sync(1, Decimal("-0.4"))
    snapshot = ledger.snapshot()
    assert snapshot.reserved_usd == Decimal("0.6")
    assert snapshot.remaining_usd == Decimal("0.4")


def test_settle_rejects_a_negative_estimate_without_freeing_reserved() -> None:
    ledger = BudgetLedger(money_budget(), store=FakeStore())
    ledger._reserve_sync(Decimal("0.5"))
    with pytest.raises(ValueError):
        ledger._settle_sync(Decimal("-0.1"), Decimal("0.05"))
    snapshot = ledger.snapshot()
    assert snapshot.reserved_usd == Decimal("0.5")
    assert snapshot.spent_usd == Decimal("0")


def test_wall_clock_exhaustion_blocks_requests_and_scans() -> None:
    now = [0.0]
    ledger = BudgetLedger(money_budget(), store=FakeStore(), clock=lambda: now[0])
    now[0] = 61.0
    with pytest.raises(BudgetExhausted) as request_exc:
        ledger._reserve_requests_sync(1)
    assert request_exc.value.reason == "max_wall_clock_s"
    with pytest.raises(BudgetExhausted) as scan_exc:
        ledger._reserve_scan_sync(1, Decimal("0.1"))
    assert scan_exc.value.reason == "max_wall_clock_s"


def test_ledgers_for_one_run_share_a_lock() -> None:
    store = FakeStore()
    first = BudgetLedger(money_budget(), store=store, run_id="run-shared-lock")
    second = BudgetLedger(money_budget(), store=store, run_id="run-shared-lock")
    assert first._lock is second._lock


async def test_ledgers_for_one_run_serialize_persistence() -> None:
    store = SlowStore()
    first = BudgetLedger(money_budget(), store=store, run_id="run-serialized", throttle_s=0.0)
    second = BudgetLedger(money_budget(), store=store, run_id="run-serialized", throttle_s=0.0)
    await asyncio.gather(
        first.reserve(Decimal("0.1"), kind="first"),
        second.reserve(Decimal("0.1"), kind="second"),
    )
    assert store.max_active == 1
    assert len(store.saved) == 2


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
    ledger = BudgetLedger(money_budget(), store=FakeStore())
    await ledger.reserve(Decimal("0.9"), kind="test")
    ledger.update_budget(Budget(max_cost_usd=2.0, max_wall_clock_s=120.0, max_target_requests=5))
    assert ledger.snapshot().remaining_usd == Decimal("1.1")
