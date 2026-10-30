"""Build a FastAPI app backed by a temporary SQLite database."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from eva_agent.config.settings import AppConfig
from eva_agent.server.app import create_app
from eva_agent.server.passwords import hash_password
from eva_agent.server.ratelimit import LoginRateLimiter
from eva_agent.server.state import AppState
from eva_agent.storage.audit import AuditRepository
from eva_agent.storage.commands import RunCommandRepository
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
from eva_agent.storage.repository import RunRepository
from eva_agent.storage.sessions import AdminSessionRepository
from tests.support.runtime import fake_config

TEST_PASSWORD = "correct horse battery"


async def build_test_app(
    tmp_path: Path,
    *,
    password: str = TEST_PASSWORD,
    config: AppConfig | None = None,
) -> tuple[FastAPI, AppState]:
    cfg = config or fake_config(tmp_path)
    engine = create_engine(cfg.storage.url)
    await init_db(engine)
    factory = create_session_factory(engine)
    index_html = tmp_path / "index.html"
    index_html.write_text("<!doctype html><title>test</title>", encoding="utf-8")
    state = AppState(
        config=cfg,
        engine=engine,
        session_factory=factory,
        runs=RunRepository(factory),
        commands=RunCommandRepository(factory),
        sessions=AdminSessionRepository(factory),
        reports=ReportRepository(factory),
        artifacts=ArtifactRepository(factory),
        audit=AuditRepository(factory),
        admin_password_hash=hash_password(password),
        login_limiter=LoginRateLimiter(),
        index_html=index_html,
    )
    return create_app(state), state
