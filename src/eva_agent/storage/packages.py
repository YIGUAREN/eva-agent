"""Stored grader package metadata."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.storage.models import GraderPackageRow


class GraderPackageRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def add(self, *, package_id: str, run_id: str, sha256: str, size_bytes: int) -> None:
        # Delete+insert replaces the run's previous package. Two exporters can
        # both delete (no row) and then race to insert the unique ``run_id``
        # row; the loser gets ``IntegrityError``. Retry the pair once so the
        # winner's row is replaced by this run's fresh archive instead of
        # being left with a stale sha.
        for attempt in range(2):
            try:
                async with self._session_factory() as session:
                    await session.execute(
                        delete(GraderPackageRow).where(GraderPackageRow.run_id == run_id)
                    )
                    session.add(
                        GraderPackageRow(
                            id=package_id,
                            run_id=run_id,
                            sha256=sha256,
                            size_bytes=size_bytes,
                        )
                    )
                    await session.commit()
                return
            except IntegrityError:
                if attempt == 1:
                    raise

    async def latest(self, run_id: str) -> GraderPackageRow | None:
        async with self._session_factory() as session:
            return await session.scalar(
                select(GraderPackageRow).where(GraderPackageRow.run_id == run_id)
            )
