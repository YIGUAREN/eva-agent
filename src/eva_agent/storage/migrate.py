"""Run Alembic migrations for the configured database."""

from __future__ import annotations

from alembic.config import Config as AlembicConfig

from alembic import command


def run_migrations(database_url: str) -> None:
    """Upgrade the database at *database_url* to Alembic head."""
    config = AlembicConfig("alembic.ini")
    # Alembic stores options in a ConfigParser, so a literal ``%`` (percent
    # encoded credentials, for example) must be doubled before interpolation.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(config, "head")
