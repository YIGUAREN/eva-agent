from eva_agent.storage.db import create_engine, create_session_factory, init_db
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
