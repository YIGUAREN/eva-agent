"""Repositories for the budget ledger and idempotent decisions."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.budget.types import RunUsage, quantize_money
from eva_agent.storage.models import BudgetDecisionRow, RunUsageRow, utcnow


def _decimal(value: Decimal | float | int | None) -> Decimal:
    return quantize_money(Decimal(str(value or 0)))


class RunUsageRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(self, run_id: str) -> RunUsageRow | None:
        async with self._session_factory() as session:
            return await session.get(RunUsageRow, run_id)

    @staticmethod
    def to_usage(row: RunUsageRow | None) -> RunUsage:
        if row is None:
            return RunUsage()
        return RunUsage(
            spent_usd=_decimal(row.spent_usd),
            reserved_usd=_decimal(row.reserved_usd),
            spent_wall_clock_s=float(row.spent_wall_clock_s or 0.0),
            target_requests=int(row.target_requests or 0),
            aig_requests=int(row.aig_requests or 0),
        )

    async def save(self, run_id: str, usage: RunUsage) -> None:
        async with self._session_factory() as session:
            row = await session.get(RunUsageRow, run_id)
            if row is None:
                session.add(
                    RunUsageRow(
                        run_id=run_id,
                        spent_usd=usage.spent_usd,
                        reserved_usd=usage.reserved_usd,
                        spent_wall_clock_s=usage.spent_wall_clock_s,
                        target_requests=usage.target_requests,
                        aig_requests=usage.aig_requests,
                        updated_at=utcnow(),
                    )
                )
            else:
                row.spent_usd = usage.spent_usd
                row.reserved_usd = usage.reserved_usd
                row.spent_wall_clock_s = usage.spent_wall_clock_s
                row.target_requests = usage.target_requests
                row.aig_requests = usage.aig_requests
                row.updated_at = utcnow()
            await session.commit()


class SqlLedgerStore:
    def __init__(self, repository: RunUsageRepository) -> None:
        self._repository = repository

    async def load(self, run_id: str) -> RunUsage | None:
        row = await self._repository.get(run_id)
        return None if row is None else self._repository.to_usage(row)

    async def save(self, run_id: str, usage: RunUsage) -> None:
        await self._repository.save(run_id, usage)


class BudgetDecisionRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def next_seq(self, run_id: str) -> int:
        async with self._session_factory() as session:
            latest = await session.scalar(
                select(BudgetDecisionRow.seq)
                .where(BudgetDecisionRow.run_id == run_id)
                .order_by(BudgetDecisionRow.seq.desc())
                .limit(1)
            )
            return int(latest or 0) + 1

    async def record(self, run_id: str, seq: int, action: str, payload: dict[str, Any]) -> bool:
        async with self._session_factory() as session:
            session.add(
                BudgetDecisionRow(
                    run_id=run_id,
                    seq=seq,
                    action=action,
                    payload_json=payload,
                    created_at=utcnow(),
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                # Only a genuine duplicate is idempotent; other constraint
                # violations (foreign keys, etc.) must surface to the caller.
                existing = await session.get(BudgetDecisionRow, (run_id, seq))
                if existing is not None and existing.action == action and existing.payload_json == payload:
                    return False
                raise
            return True

    async def decide(self, run_id: str, seq: int, action: str, payload: dict[str, Any]) -> bool:
        """Claim the pending gate row for one decision.

        Returns True only for the single delivery that transitions the row away
        from ``gate``.  Every later delivery returns False, whether it repeats
        the same action/payload or tries a different one, and never overwrites
        the recorded decision: the first decision for a seq is final.
        """
        async with self._session_factory() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(BudgetDecisionRow)
                    .where(
                        BudgetDecisionRow.run_id == run_id,
                        BudgetDecisionRow.seq == seq,
                        BudgetDecisionRow.action == "gate",
                    )
                    .values(action=action, payload_json=payload)
                ),
            )
            await session.commit()
            return bool(result.rowcount)

    async def get(self, run_id: str, seq: int) -> BudgetDecisionRow | None:
        async with self._session_factory() as session:
            return await session.get(BudgetDecisionRow, (run_id, seq))

    async def latest_gate(self, run_id: str) -> BudgetDecisionRow | None:
        async with self._session_factory() as session:
            return await session.scalar(
                select(BudgetDecisionRow)
                .where(BudgetDecisionRow.run_id == run_id, BudgetDecisionRow.action == "gate")
                .order_by(BudgetDecisionRow.seq.desc())
                .limit(1)
            )

    async def latest_decision(self, run_id: str) -> BudgetDecisionRow | None:
        """Return the most recent row that already carries an operator decision."""
        async with self._session_factory() as session:
            return await session.scalar(
                select(BudgetDecisionRow)
                .where(BudgetDecisionRow.run_id == run_id, BudgetDecisionRow.action != "gate")
                .order_by(BudgetDecisionRow.seq.desc())
                .limit(1)
            )

    async def list_for_run(self, run_id: str) -> list[BudgetDecisionRow]:
        async with self._session_factory() as session:
            return list(
                await session.scalars(
                    select(BudgetDecisionRow)
                    .where(BudgetDecisionRow.run_id == run_id)
                    .order_by(BudgetDecisionRow.seq)
                )
            )
