"""Server-side admin sessions."""

from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.storage.models import AdminSessionRow, utcnow


def _aware(value: datetime) -> datetime:
    """SQLite returns naive datetimes; treat stored values as UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class AdminSessionRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(
        self,
        id_hash: str,
        csrf_token: str,
        expires_at: datetime,
        absolute_expires_at: datetime,
    ) -> None:
        async with self._session_factory() as session:
            session.add(
                AdminSessionRow(
                    id_hash=id_hash,
                    csrf_token=csrf_token,
                    expires_at=expires_at,
                    absolute_expires_at=absolute_expires_at,
                )
            )
            await session.commit()

    async def get(self, id_hash: str, *, now: datetime | None = None) -> AdminSessionRow | None:
        moment = now or utcnow()
        async with self._session_factory() as session:
            row = await session.get(AdminSessionRow, id_hash)
            if row is None or row.revoked_at is not None:
                return None
            if _aware(row.expires_at) <= moment or _aware(row.absolute_expires_at) <= moment:
                return None
            return row

    async def touch(self, id_hash: str, *, expires_at: datetime, now: datetime) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(AdminSessionRow)
                .where(AdminSessionRow.id_hash == id_hash)
                .values(last_seen_at=now, expires_at=expires_at)
            )
            await session.commit()

    async def revoke(self, id_hash: str) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(AdminSessionRow)
                .where(AdminSessionRow.id_hash == id_hash)
                .values(revoked_at=utcnow())
            )
            await session.commit()
