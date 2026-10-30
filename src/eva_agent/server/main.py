"""eva-web entrypoint: API plus the built frontend."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import uvicorn
from alembic.config import Config as AlembicConfig

from alembic import command as alembic_command
from eva_agent.config.settings import load_config
from eva_agent.server.app import create_app
from eva_agent.server.ratelimit import LoginRateLimiter
from eva_agent.server.state import AppState
from eva_agent.storage.audit import AuditRepository
from eva_agent.storage.commands import RunCommandRepository
from eva_agent.storage.db import create_engine, create_session_factory
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
from eva_agent.storage.repository import RunRepository
from eva_agent.storage.sessions import AdminSessionRepository

DEFAULT_INDEX = Path("web/dist/index.html")


def _migrate(database_url: str) -> None:
    config = AlembicConfig("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    alembic_command.upgrade(config, "head")


async def serve(config_path: Path, host: str, port: int) -> None:
    config = load_config(config_path)
    password_hash = os.environ.get("ADMIN_PASSWORD_HASH", "")
    if not password_hash:
        raise SystemExit("ADMIN_PASSWORD_HASH is not set; run python scripts/hash_password.py")

    await asyncio.to_thread(_migrate, config.storage.url)

    engine = create_engine(config.storage.url)
    factory = create_session_factory(engine)
    state = AppState(
        config=config,
        engine=engine,
        session_factory=factory,
        runs=RunRepository(factory),
        commands=RunCommandRepository(factory),
        sessions=AdminSessionRepository(factory),
        reports=ReportRepository(factory),
        artifacts=ArtifactRepository(factory),
        audit=AuditRepository(factory),
        admin_password_hash=password_hash,
        login_limiter=LoginRateLimiter(),
        index_html=DEFAULT_INDEX,
    )
    app = create_app(state)
    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="info", proxy_headers=True, forwarded_allow_ips="127.0.0.1")
    )
    try:
        await server.serve()
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="eva-web")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(serve(Path(args.config), args.host, args.port))


if __name__ == "__main__":
    main()
