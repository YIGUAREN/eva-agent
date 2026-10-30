from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from alembic import command
from eva_agent.storage.models import Base

_ALEMBIC_ROOT = Path(__file__).resolve().parents[2]


def test_upgrade_head_matches_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    url = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
    config = Config(str(_ALEMBIC_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_ALEMBIC_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url)

    command.upgrade(config, "head")

    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        assert set(inspector.get_table_names()) >= set(Base.metadata.tables)
        for table in Base.metadata.sorted_tables:
            actual = {column["name"] for column in inspector.get_columns(table.name)}
            assert actual == {column.name for column in table.columns}, table.name

        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={"compare_type": True, "compare_server_default": True},
            )
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()
