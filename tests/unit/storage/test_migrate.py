"""Tests for the shared Alembic migration helper."""

from pathlib import Path

from sqlalchemy import create_engine, inspect

from eva_agent.storage.migrate import run_migrations


def test_run_migrations_is_idempotent_and_escapes_percent(tmp_path: Path) -> None:
    # ``%40`` exercises configparser interpolation: an unescaped percent sign
    # would crash before Alembic ever reaches the database.
    url = f"sqlite:///{(tmp_path / 'migrate%40db.sqlite').as_posix()}"

    run_migrations(url)
    run_migrations(url)

    engine = create_engine(url)
    try:
        assert "runs" in inspect(engine).get_table_names()
    finally:
        engine.dispose()
