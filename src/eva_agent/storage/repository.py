"""Repository for run state and events."""

from datetime import datetime
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.run import RunInput, RunStatus
from eva_agent.storage.models import RunEventCounterRow, RunEventRow, RunRow, utcnow


class RunRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(
        self,
        run_id: str,
        run_input: RunInput,
        budget: Budget,
        *,
        idempotency_key: str | None = None,
        authorization_ref: str | None = None,
        created_by: str | None = None,
    ) -> None:
        async with self._session_factory() as session:
            session.add(
                RunRow(
                    id=run_id,
                    status=RunStatus.QUEUED.value,
                    input_json=run_input.model_dump(mode="json"),
                    budget_json=budget.model_dump(mode="json"),
                    idempotency_key=idempotency_key,
                    authorization_ref=authorization_ref,
                    created_by=created_by,
                )
            )
            await session.flush()
            session.add(RunEventCounterRow(run_id=run_id, next_seq=0))
            await session.commit()

    async def get(self, run_id: str) -> RunRow | None:
        async with self._session_factory() as session:
            return await session.get(RunRow, run_id)

    async def get_by_idempotency_key(self, key: str) -> RunRow | None:
        async with self._session_factory() as session:
            return await session.scalar(select(RunRow).where(RunRow.idempotency_key == key))

    async def set_status(self, run_id: str, status: RunStatus) -> None:
        async with self._session_factory() as session:
            row = await session.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.status = status.value
            await session.commit()

    async def claim(
        self,
        run_id: str,
        *,
        from_statuses: tuple[RunStatus, ...],
        to_status: RunStatus,
    ) -> bool:
        """Atomically move a run between statuses; report whether the row changed."""
        async with self._session_factory() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(RunRow)
                    .where(
                        RunRow.id == run_id,
                        RunRow.status.in_([status.value for status in from_statuses]),
                    )
                    .values(status=to_status.value)
                ),
            )
            await session.commit()
            return bool(result.rowcount)

    async def set_error(self, run_id: str, error: str) -> None:
        async with self._session_factory() as session:
            row = await session.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.error = error
            await session.commit()

    async def set_public(self, run_id: str, public: bool, *, now: datetime | None = None) -> RunRow:
        async with self._session_factory() as session:
            row = await session.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.is_public = public
            row.published_at = (now or utcnow()) if public else None
            await session.commit()
            return row

    async def list_public(
        self,
        *,
        cursor: tuple[datetime, str] | None,
        limit: int,
    ) -> list[RunRow]:
        query = select(RunRow).where(RunRow.is_public.is_(True))
        if cursor is not None:
            created_at, run_id = cursor
            query = query.where(
                (RunRow.created_at < created_at) | ((RunRow.created_at == created_at) & (RunRow.id < run_id))
            )
        query = query.order_by(RunRow.created_at.desc(), RunRow.id.desc()).limit(limit)
        async with self._session_factory() as session:
            return list(await session.scalars(query))

    async def list_by_status(self, status: RunStatus) -> list[RunRow]:
        async with self._session_factory() as session:
            return list(await session.scalars(select(RunRow).where(RunRow.status == status.value)))

    async def append_event(self, run_id: str, event_type: str, payload: dict | None = None) -> int:
        async with self._session_factory() as session:
            seq = await session.scalar(
                update(RunEventCounterRow)
                .where(RunEventCounterRow.run_id == run_id)
                .values(next_seq=RunEventCounterRow.next_seq + 1)
                .returning(RunEventCounterRow.next_seq)
            )
            if seq is None:
                raise KeyError(run_id)
            session.add(RunEventRow(run_id=run_id, seq=seq, type=event_type, payload_json=payload or {}))
            await session.commit()
            return seq

    async def list_events(self, run_id: str, *, after_seq: int = 0) -> list[RunEventRow]:
        async with self._session_factory() as session:
            rows = await session.scalars(
                select(RunEventRow)
                .where(RunEventRow.run_id == run_id, RunEventRow.seq > after_seq)
                .order_by(RunEventRow.seq)
            )
            return list(rows)

    async def latest_event_seq(self, run_id: str) -> int:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(RunEventRow.seq).where(RunEventRow.run_id == run_id).order_by(RunEventRow.seq.desc()).limit(1)
            )
            return int(row or 0)

    async def set_plan(self, run_id: str, plan: EvalPlan) -> None:
        async with self._session_factory() as session:
            row = await session.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.plan_json = plan.model_dump(mode="json")
            await session.commit()

    async def set_budget(self, run_id: str, budget: Budget) -> None:
        async with self._session_factory() as session:
            row = await session.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.budget_json = budget.model_dump(mode="json")
            await session.commit()
