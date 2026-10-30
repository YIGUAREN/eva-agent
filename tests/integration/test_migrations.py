from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command
from eva_agent.storage.models import Base


def test_upgrade_head_matches_metadata(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
    config = Config("alembic.ini")
    config.set_main_option("script_location", "alembic")
    config.set_main_option("sqlalchemy.url", url)

    command.upgrade(config, "head")

    engine = create_engine(url)
    inspector = inspect(engine)
    assert set(inspector.get_table_names()) >= set(Base.metadata.tables)
    for table in Base.metadata.sorted_tables:
        actual = {column["name"] for column in inspector.get_columns(table.name)}
        assert actual == {column.name for column in table.columns}, table.name
    engine.dispose()
