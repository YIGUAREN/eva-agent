from datetime import UTC, datetime, timedelta

from sqlalchemy import String

from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import UploadRow
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


def test_upload_id_column_accepts_64_characters() -> None:
    # PostgreSQL enforces VARCHAR length; generated ids are ``upload_<32 hex>``
    # (39 chars) and the validator allows up to 64, so 36 would reject every row.
    column_type = UploadRow.__table__.c.id.type
    assert isinstance(column_type, String)
    assert column_type.length == 64


async def test_upload_id_round_trips_at_maximum_lengths(tmp_path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'u.db').as_posix()}")
    await init_db(engine)
    repo = UploadRepository(create_session_factory(engine))
    expires = datetime.now(UTC) + timedelta(days=30)

    generated = "upload_" + "a" * 32  # 39 characters
    longest = "u" * 64  # maximum allowed by validate_upload_id

    for upload_id in (generated, longest):
        await repo.create(
            upload_id,
            filename="pkg.zip",
            size_bytes=12,
            sha256="b" * 64,
            format="zip",
            expires_at=expires,
        )
        row = await repo.get(upload_id)
        assert row is not None
        assert row.id == upload_id
