from datetime import UTC, datetime, timedelta

from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.uploads import (
    UPLOAD_EXTRACTED,
    UPLOAD_EXTRACTING,
    UPLOAD_FAILED,
    UploadRepository,
)


async def test_upload_lifecycle(tmp_path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'u.db').as_posix()}")
    await init_db(engine)
    repo = UploadRepository(create_session_factory(engine))
    expires = datetime.now(UTC) + timedelta(days=30)

    await repo.create(
        "upload_abc",
        filename="pkg.zip",
        size_bytes=12,
        sha256="a" * 64,
        format="zip",
        expires_at=expires,
    )
    row = await repo.get("upload_abc")
    assert row is not None
    assert row.status == UPLOAD_EXTRACTING

    await repo.mark_extracted("upload_abc", {"format": "zip", "files": [], "total_bytes": 0})
    row = await repo.get("upload_abc")
    assert row is not None
    assert row.status == UPLOAD_EXTRACTED
    assert row.manifest == {"format": "zip", "files": [], "total_bytes": 0}

    await repo.mark_failed("upload_abc", "corrupt")
    row = await repo.get("upload_abc")
    assert row is not None
    assert row.status == UPLOAD_FAILED
    assert row.error == "corrupt"


async def test_get_unknown_upload_returns_none(tmp_path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'u.db').as_posix()}")
    await init_db(engine)
    repo = UploadRepository(create_session_factory(engine))
    assert await repo.get("upload_missing") is None
