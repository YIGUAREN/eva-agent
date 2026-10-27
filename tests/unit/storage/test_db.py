from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import RunEventRow


async def test_init_db_creates_missing_sqlite_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "missing" / "nested" / "eva.db"
    assert not db_path.parent.exists()

    engine = create_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    try:
        await init_db(engine)
    finally:
        await engine.dispose()

    assert db_path.is_file()


async def test_sqlite_foreign_keys_are_enforced(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}")
    await init_db(engine)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            session.add(RunEventRow(run_id="missing", seq=1, type="run.status", payload_json={}))
            with pytest.raises(IntegrityError):
                await session.commit()
    finally:
        await engine.dispose()
