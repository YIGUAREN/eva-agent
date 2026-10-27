"""Async engine helpers."""

from pathlib import Path

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from eva_agent.storage.models import Base


def _ensure_sqlite_parent_directory(url: str) -> None:
    """Create the parent directory for file-backed SQLite URLs."""
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        return
    database = parsed.database
    if not database or database == ":memory:":
        return
    Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)


def create_engine(url: str) -> AsyncEngine:
    _ensure_sqlite_parent_directory(url)
    return create_async_engine(url)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Create tables. Alembic replaces this in a later phase."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
