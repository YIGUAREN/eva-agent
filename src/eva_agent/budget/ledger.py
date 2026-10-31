"""Per-run ledger: reserve before a call, settle after it, persist on a throttle."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
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
        self._budget = budget
        self._store = store
        self._run_id = run_id
        self._unlimited = unlimited
        self._clock = clock
        self._throttle_s = throttle_s
        self._state = state or RunUsage()
        self._started = clock()
        self._last_persist = clock()
        self._dirty = False
        self._estimated = False
        self._lock = asyncio.Lock()

    @classmethod
    def unlimited(cls, budget: Budget) -> BudgetLedger:
        return cls(budget, unlimited=True)

    @property
    def enforced(self) -> bool:
        return not self._unlimited and self._store is not None

    @property
    def estimated(self) -> bool:
        return self._estimated

    def update_budget(self, budget: Budget) -> None:
        self._budget = budget

    def _elapsed(self) -> float:
        return self._state.spent_wall_clock_s + max(0.0, self._clock() - self._started)

    def _remaining(self) -> Decimal:
        if self._unlimited:
            return Decimal("Infinity")
        return max(
            ZERO,
            quantize_money(Decimal(str(self._budget.max_cost_usd)))
            - self._state.spent_usd
            - self._state.reserved_usd,
        )

    def _check_cost(self, estimate: Decimal) -> None:
        if self._unlimited:
            return
        if self._state.spent_usd + self._state.reserved_usd + estimate > quantize_money(
            Decimal(str(self._budget.max_cost_usd))
        ):
            raise BudgetExhausted(
                spent=self._state.spent_usd,
                remaining=self._remaining(),
                reason="max_cost_usd",
            )

    def _check_wall_clock(self) -> None:
        if self._unlimited:
            return
        if self._elapsed() > self._budget.max_wall_clock_s:
            raise BudgetExhausted(
                spent=self._state.spent_usd,
                remaining=self._remaining(),
                reason="max_wall_clock_s",
            )

    def _reserve_sync(self, estimate: Decimal) -> None:
        estimate = quantize_money(estimate)
        if estimate < ZERO:
            raise ValueError("estimate must be non-negative")
        self._check_wall_clock()
        self._check_cost(estimate)
        self._state = RunUsage(
            spent_usd=self._state.spent_usd,
            reserved_usd=self._state.reserved_usd + estimate,
            spent_wall_clock_s=self._elapsed(),
            target_requests=self._state.target_requests,
            aig_requests=self._state.aig_requests,
        )
        self._touch()

    def _settle_sync(self, estimate: Decimal, actual: Decimal | None) -> None:
        estimate = quantize_money(estimate)
        spent = estimate if actual is None else quantize_money(actual)
        if spent < ZERO:
            raise ValueError("actual must be non-negative")
        reserved = max(ZERO, self._state.reserved_usd - estimate)
        self._state = RunUsage(
            spent_usd=self._state.spent_usd + spent,
            reserved_usd=reserved,
            spent_wall_clock_s=self._elapsed(),
            target_requests=self._state.target_requests,
            aig_requests=self._state.aig_requests,
        )
        if actual is None:
            self._estimated = True
        self._touch()

    def _reserve_requests_sync(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if not self._unlimited and self._state.target_requests + count > self._budget.max_target_requests:
            raise BudgetExhausted(
                spent=self._state.spent_usd,
                remaining=self._remaining(),
                reason="max_target_requests",
            )
        self._state = RunUsage(
            spent_usd=self._state.spent_usd,
            reserved_usd=self._state.reserved_usd,
            spent_wall_clock_s=self._elapsed(),
            target_requests=self._state.target_requests + count,
            aig_requests=self._state.aig_requests,
        )
        self._touch()

    def _reserve_scan_sync(self, count: int, estimate: Decimal) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        # AIG scans hit the target too, so they share the max_target_requests pool.
        if not self._unlimited and self._state.target_requests + count > self._budget.max_target_requests:
            raise BudgetExhausted(
                spent=self._state.spent_usd,
                remaining=self._remaining(),
                reason="max_target_requests",
            )
        estimate = quantize_money(estimate)
        self._check_cost(estimate)
        self._state = RunUsage(
            spent_usd=self._state.spent_usd,
            reserved_usd=self._state.reserved_usd + estimate,
            spent_wall_clock_s=self._elapsed(),
            target_requests=self._state.target_requests + count,
            aig_requests=self._state.aig_requests + count,
        )
        self._touch()

    def _touch(self) -> None:
        self._dirty = True

    async def reserve(self, estimate: Decimal, *, kind: str) -> None:
        del kind
        async with self._lock:
            self._reserve_sync(estimate)
            await self._persist()

    async def settle(self, estimate: Decimal, actual: Decimal | None, *, kind: str) -> None:
        del kind
        async with self._lock:
            self._settle_sync(estimate, actual)
            await self._persist()

    async def reserve_requests(self, n: int) -> None:
        async with self._lock:
            self._reserve_requests_sync(n)
            await self._persist()

    async def reserve_scan(self, count: int, estimate: Decimal) -> None:
        async with self._lock:
            self._reserve_scan_sync(count, estimate)
            await self._persist()

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            spent_usd=self._state.spent_usd,
            reserved_usd=self._state.reserved_usd,
            remaining_usd=self._remaining(),
            spent_wall_clock_s=self._elapsed(),
            target_requests=self._state.target_requests,
            aig_requests=self._state.aig_requests,
            estimated=self._estimated,
        )

    async def _persist(self, *, force: bool = False) -> None:
        if self._store is None or self._run_id == "":
            return
        now = self._clock()
        if not force and (not self._dirty or now - self._last_persist < self._throttle_s):
            return
        state = RunUsage(
            spent_usd=self._state.spent_usd,
            reserved_usd=self._state.reserved_usd,
            spent_wall_clock_s=self._elapsed(),
            target_requests=self._state.target_requests,
            aig_requests=self._state.aig_requests,
        )
        await self._store.save(self._run_id, state)
        self._last_persist = now
        self._dirty = False

    async def flush(self) -> None:
        async with self._lock:
            await self._persist(force=True)
