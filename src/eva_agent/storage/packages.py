"""Stored grader package metadata."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.storage.models import GraderPackageRow


class GraderPackageRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def add(self, *, package_id: str, run_id: str, sha256: str, size_bytes: int) -> None:
        async with self._session_factory() as session:
            await session.execute(delete(GraderPackageRow).where(GraderPackageRow.run_id == run_id))
            session.add(
                GraderPackageRow(
                    id=package_id,
                    run_id=run_id,
                    sha256=sha256,
                    size_bytes=size_bytes,
                )
            )
            await session.commit()

    async def latest(self, run_id: str) -> GraderPackageRow | None:
        async with self._session_factory() as session:
            return await session.scalar(
                select(GraderPackageRow).where(GraderPackageRow.run_id == run_id)
            )
