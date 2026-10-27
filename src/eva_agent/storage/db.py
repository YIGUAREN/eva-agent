"""Async engine helpers."""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from eva_agent.storage.models import Base


def create_engine(url: str) -> AsyncEngine:
    return create_async_engine(url)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Create tables. Alembic replaces this in a later phase."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
