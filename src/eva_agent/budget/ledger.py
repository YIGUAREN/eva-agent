"""Per-run ledger: reserve before a call, settle after it, persist on a throttle."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

from eva_agent.budget.types import (
    BudgetExhausted,
    BudgetSnapshot,
    LedgerStore,
    RunUsage,
    quantize_money,
)
from eva_agent.domain.budget import Budget

ZERO = Decimal("0")

# Locks live on the event loop that uses them, so a fresh ``asyncio.run`` never
# inherits a lock bound to a previous loop.
_RUN_LOCKS_ATTR = "_eva_budget_run_locks"


@dataclass
class _RunState:
    """Mutable state shared by every enforced ledger for one run id."""

    budget: Budget
    usage: RunUsage = field(default_factory=RunUsage)
    started: float = 0.0
    last_persist: float = 0.0
    dirty: bool = False
    estimated: bool = False


# Process-level state per run.  Every enforced ledger with the same non-empty
# run_id reads and writes this entry, so one instance can never persist a stale
# snapshot over another instance's update.
_RUN_STATES: dict[str, _RunState] = {}


def _lock_for(run_id: str) -> asyncio.Lock:
    """Return the lock for this run on the current event loop, creating it lazily."""
    loop = asyncio.get_running_loop()
    locks = getattr(loop, _RUN_LOCKS_ATTR, None)
    if not isinstance(locks, dict):
        locks = {}
        setattr(loop, _RUN_LOCKS_ATTR, locks)
    lock: asyncio.Lock | None = locks.get(run_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[run_id] = lock
    return lock


class BudgetLedger:
    def __init__(
        self,
        budget: Budget,
        *,
        store: LedgerStore | None = None,
        run_id: str = "",
        unlimited: bool = False,
        clock: Callable[[], float] = time.monotonic,
        throttle_s: float = 1.0,
        state: RunUsage | None = None,
    ) -> None:
        """Create a ledger for one run.

        ``state`` only seeds the first ledger created for a non-empty ``run_id``.
        Later ledgers for the same run join the live in-memory entry (adopting
        its usage, elapsed baseline, and budget limits) and ignore their seed.
        """
        if not unlimited and store is None:
            # Fail closed: an enforced ledger without a store would let callers
            # believe budget checks are active while persistence silently no-ops.
            raise ValueError("a limited BudgetLedger requires a store")
        self._store = store
        self._run_id = run_id
        self._unlimited = unlimited
        self._clock = clock
        self._throttle_s = throttle_s
        started = clock()
        if not unlimited and run_id != "":
            shared = _RUN_STATES.get(run_id)
            if shared is None:
                shared = _RunState(
                    budget=budget,
                    usage=state or RunUsage(),
                    started=started,
                    last_persist=started,
                )
                _RUN_STATES[run_id] = shared
            # A later ledger joins the existing run state; it must not reload
            # over it, reset the elapsed baseline, or impose its own limits.
            self._run_state = shared
        else:
            self._run_state = _RunState(
                budget=budget,
                usage=state or RunUsage(),
                started=started,
                last_persist=started,
            )

    @classmethod
    def unlimited(cls, budget: Budget) -> BudgetLedger:
        return cls(budget, unlimited=True)

    @property
    def enforced(self) -> bool:
        return not self._unlimited

    @property
    def estimated(self) -> bool:
        return self._run_state.estimated

    def update_budget(self, budget: Budget) -> None:
        # Limits are run-level: an extension must apply to every ledger that
        # shares this run's state, not just the one that received the update.
        self._run_state.budget = budget

    def _elapsed(self) -> float:
        # The usage stores the wall clock loaded from the store at construction;
        # mutators must leave it untouched or the baseline is re-added each time.
        return self._run_state.usage.spent_wall_clock_s + max(0.0, self._clock() - self._run_state.started)

    def _remaining(self) -> Decimal:
        usage = self._run_state.usage
        if self._unlimited:
            return Decimal("Infinity")
        return max(
            ZERO,
            quantize_money(Decimal(str(self._run_state.budget.max_cost_usd))) - usage.spent_usd - usage.reserved_usd,
        )

    def _check_cost(self, estimate: Decimal) -> None:
        usage = self._run_state.usage
        if self._unlimited:
            return
        limit = quantize_money(Decimal(str(self._run_state.budget.max_cost_usd)))
        if usage.spent_usd + usage.reserved_usd + estimate > limit:
            raise BudgetExhausted(
                spent=usage.spent_usd,
                remaining=self._remaining(),
                reason="max_cost_usd",
            )

    def _check_wall_clock(self) -> None:
        if self._unlimited:
            return
        if self._elapsed() > self._run_state.budget.max_wall_clock_s:
            raise BudgetExhausted(
                spent=self._run_state.usage.spent_usd,
                remaining=self._remaining(),
                reason="max_wall_clock_s",
            )

    def _reserve_sync(self, estimate: Decimal) -> None:
        estimate = quantize_money(estimate)
        if estimate < ZERO:
            raise ValueError("estimate must be non-negative")
        self._check_wall_clock()
        self._check_cost(estimate)
        usage = self._run_state.usage
        self._run_state.usage = RunUsage(
            spent_usd=usage.spent_usd,
            reserved_usd=usage.reserved_usd + estimate,
            spent_wall_clock_s=usage.spent_wall_clock_s,
            target_requests=usage.target_requests,
            aig_requests=usage.aig_requests,
        )
        self._touch()

    def _settle_sync(self, estimate: Decimal, actual: Decimal | None) -> None:
        estimate = quantize_money(estimate)
        if estimate < ZERO:
            raise ValueError("estimate must be non-negative")
        spent = estimate if actual is None else quantize_money(actual)
        if spent < ZERO:
            raise ValueError("actual must be non-negative")
        usage = self._run_state.usage
        reserved = max(ZERO, usage.reserved_usd - estimate)
        self._run_state.usage = RunUsage(
            spent_usd=usage.spent_usd + spent,
            reserved_usd=reserved,
            spent_wall_clock_s=usage.spent_wall_clock_s,
            target_requests=usage.target_requests,
            aig_requests=usage.aig_requests,
        )
        if actual is None:
            self._run_state.estimated = True
        self._touch()

    def _reserve_requests_sync(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._check_wall_clock()
        usage = self._run_state.usage
        if not self._unlimited and usage.target_requests + count > self._run_state.budget.max_target_requests:
            raise BudgetExhausted(
                spent=usage.spent_usd,
                remaining=self._remaining(),
                reason="max_target_requests",
            )
        self._run_state.usage = RunUsage(
            spent_usd=usage.spent_usd,
            reserved_usd=usage.reserved_usd,
            spent_wall_clock_s=usage.spent_wall_clock_s,
            target_requests=usage.target_requests + count,
            aig_requests=usage.aig_requests,
        )
        self._touch()

    def _reserve_scan_sync(self, count: int, estimate: Decimal) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        estimate = quantize_money(estimate)
        if estimate < ZERO:
            raise ValueError("estimate must be non-negative")
        self._check_wall_clock()
        usage = self._run_state.usage
        # AIG scans hit the target too, so they share the max_target_requests pool.
        if not self._unlimited and usage.target_requests + count > self._run_state.budget.max_target_requests:
            raise BudgetExhausted(
                spent=usage.spent_usd,
                remaining=self._remaining(),
                reason="max_target_requests",
            )
        self._check_cost(estimate)
        self._run_state.usage = RunUsage(
            spent_usd=usage.spent_usd,
            reserved_usd=usage.reserved_usd + estimate,
            spent_wall_clock_s=usage.spent_wall_clock_s,
            target_requests=usage.target_requests + count,
            aig_requests=usage.aig_requests + count,
        )
        self._touch()

    def _touch(self) -> None:
        self._run_state.dirty = True

    async def reserve(self, estimate: Decimal, *, kind: str) -> None:
        del kind
        async with _lock_for(self._run_id):
            self._reserve_sync(estimate)
            await self._persist()

    async def settle(self, estimate: Decimal, actual: Decimal | None, *, kind: str) -> None:
        del kind
        async with _lock_for(self._run_id):
            self._settle_sync(estimate, actual)
            await self._persist()

    async def reserve_requests(self, n: int) -> None:
        async with _lock_for(self._run_id):
            self._reserve_requests_sync(n)
            await self._persist()

    async def reserve_scan(self, count: int, estimate: Decimal) -> None:
        async with _lock_for(self._run_id):
            self._reserve_scan_sync(count, estimate)
            await self._persist()

    def snapshot(self) -> BudgetSnapshot:
        usage = self._run_state.usage
        return BudgetSnapshot(
            spent_usd=usage.spent_usd,
            reserved_usd=usage.reserved_usd,
            remaining_usd=self._remaining(),
            spent_wall_clock_s=self._elapsed(),
            target_requests=usage.target_requests,
            aig_requests=usage.aig_requests,
            estimated=self._run_state.estimated,
        )

    async def _persist(self, *, force: bool = False) -> None:
        # An unlimited ledger tracks usage locally but must never overwrite the
        # enforced ledger's row for the same run.
        if self._store is None or self._run_id == "" or self._unlimited:
            return
        now = self._clock()
        run_state = self._run_state
        if not force and (not run_state.dirty or now - run_state.last_persist < self._throttle_s):
            return
        usage = run_state.usage
        state = RunUsage(
            spent_usd=usage.spent_usd,
            reserved_usd=usage.reserved_usd,
            spent_wall_clock_s=self._elapsed(),
            target_requests=usage.target_requests,
            aig_requests=usage.aig_requests,
        )
        await self._store.save(self._run_id, state)
        run_state.last_persist = now
        run_state.dirty = False

    async def flush(self) -> None:
        async with _lock_for(self._run_id):
            await self._persist(force=True)
