"""Command queue between the web API and the worker."""

from datetime import datetime
from typing import Any, cast

from sqlalchemy import CursorResult, case, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.storage.models import RunCommandRow, utcnow


class ActiveCommandExists(RuntimeError):
    """Raised when a run already has a pending or claimed non-cancel command."""


class RunCommandRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def enqueue(self, run_id: str, kind: str, payload: dict | None = None) -> int:
        async with self._session_factory() as session:
            row = RunCommandRow(run_id=run_id, kind=kind, payload_json=payload or {})
            session.add(row)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise ActiveCommandExists(kind) from exc
            return row.id

    async def claim_next(self) -> RunCommandRow | None:
        async with self._session_factory() as session:
            next_id = (
                select(RunCommandRow.id)
                .where(RunCommandRow.status == "pending")
                .order_by(case((RunCommandRow.kind == "cancel", 0), else_=1), RunCommandRow.id)
                .limit(1)
                .scalar_subquery()
            )
            command_id = await session.scalar(
                update(RunCommandRow)
                .where(RunCommandRow.id == next_id, RunCommandRow.status == "pending")
                .values(status="claimed", claimed_at=utcnow(), attempts=RunCommandRow.attempts + 1)
                .returning(RunCommandRow.id)
            )
            if command_id is None:
                return None
            await session.commit()
            return await session.get(RunCommandRow, command_id)

    async def finish(self, command_id: int, status: str, error: str | None = None) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(RunCommandRow)
                .where(RunCommandRow.id == command_id)
                .values(status=status, error=error, finished_at=utcnow())
            )
            await session.commit()

    async def reset_stale(self, older_than: datetime) -> int:
        async with self._session_factory() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(RunCommandRow)
                    .where(RunCommandRow.status == "claimed", RunCommandRow.claimed_at < older_than)
                    .values(status="pending", claimed_at=None)
                ),
            )
            await session.commit()
            return int(result.rowcount or 0)

    async def get(self, command_id: int) -> RunCommandRow | None:
        async with self._session_factory() as session:
            return await session.get(RunCommandRow, command_id)

    async def list_for_run(self, run_id: str) -> list[RunCommandRow]:
        async with self._session_factory() as session:
            rows = await session.scalars(
                select(RunCommandRow).where(RunCommandRow.run_id == run_id).order_by(RunCommandRow.id)
            )
            return list(rows)
