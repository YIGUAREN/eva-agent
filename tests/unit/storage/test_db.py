from pathlib import Path

from eva_agent.storage.db import create_engine, init_db


async def test_init_db_creates_missing_sqlite_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "missing" / "nested" / "eva.db"
    assert not db_path.parent.exists()

    engine = create_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    try:
        await init_db(engine)
    finally:
        await engine.dispose()

    assert db_path.is_file()
