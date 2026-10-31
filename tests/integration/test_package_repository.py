from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import GraderPackageRow
from eva_agent.storage.packages import GraderPackageRepository
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import budget, fake_config, paste_input


async def test_add_then_latest_returns_the_replacement(tmp_path) -> None:
    config = fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    try:
        factory = create_session_factory(engine)
        await RunRepository(factory).create("run-1", paste_input(), budget())
        repo = GraderPackageRepository(factory)
        assert await repo.latest("run-1") is None
        await repo.add(package_id="pkg-1", run_id="run-1", sha256="a" * 64, size_bytes=10)
        await repo.add(package_id="pkg-2", run_id="run-1", sha256="b" * 64, size_bytes=11)
        row = await repo.latest("run-1")
        assert row is not None
        assert (row.id, row.sha256, row.size_bytes) == ("pkg-2", "b" * 64, 11)
    finally:
        await engine.dispose()


async def test_add_retries_once_after_a_concurrent_insert(tmp_path, monkeypatch) -> None:
    config = fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    try:
        factory = create_session_factory(engine)
        await RunRepository(factory).create("run-1", paste_input(), budget())
        repo = GraderPackageRepository(factory)
        await repo.add(package_id="pkg-old", run_id="run-1", sha256="a" * 64, size_bytes=10)

        real_commit = AsyncSession.commit
        commits = 0

        async def flaky_commit(self) -> None:
            # The first delete+insert loses a race: another exporter committed
            # the unique run_id row between our delete and our commit.
            nonlocal commits
            commits += 1
            if commits == 1:
                raise IntegrityError("INSERT", {}, Exception("UNIQUE constraint failed"))
            await real_commit(self)

        monkeypatch.setattr(AsyncSession, "commit", flaky_commit)
        await repo.add(package_id="pkg-new", run_id="run-1", sha256="b" * 64, size_bytes=11)

        assert commits == 2
        row = await repo.latest("run-1")
        assert row is not None
        assert (row.id, row.sha256, row.size_bytes) == ("pkg-new", "b" * 64, 11)
        async with factory() as session:
            count = await session.scalar(select(func.count()).select_from(GraderPackageRow))
        assert count == 1
    finally:
        await engine.dispose()
