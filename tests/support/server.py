"""Build a FastAPI app backed by a temporary SQLite database."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport

from eva_agent.config.settings import AppConfig
from eva_agent.sandbox.app import create_sandbox_app
from eva_agent.server.app import create_app
from eva_agent.server.passwords import hash_password
from eva_agent.server.ratelimit import LoginRateLimiter
from eva_agent.server.sandbox import SandboxClient
from eva_agent.server.state import AppState
from eva_agent.storage.audit import AuditRepository
from eva_agent.storage.budget import BudgetDecisionRepository, RunUsageRepository
from eva_agent.storage.commands import RunCommandRepository
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.packages import GraderPackageRepository
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
from eva_agent.storage.repository import RunRepository
from eva_agent.storage.sessions import AdminSessionRepository
from eva_agent.storage.uploads import UploadRepository
from tests.support.runtime import fake_config

TEST_PASSWORD = "correct horse battery"
SANDBOX_TOKEN = "test-sandbox-token"


async def build_test_app(
    tmp_path: Path,
    *,
    password: str = TEST_PASSWORD,
    config: AppConfig | None = None,
    sandbox: SandboxClient | None = None,
) -> tuple[FastAPI, AppState]:
    cfg = config or fake_config(tmp_path)
    engine = create_engine(cfg.storage.url)
    await init_db(engine)
    factory = create_session_factory(engine)
    index_html = tmp_path / "index.html"
    index_html.write_text("<!doctype html><title>test</title>", encoding="utf-8")
    sandbox_client = sandbox or SandboxClient(
        "http://sandbox",
        SANDBOX_TOKEN,
        5.0,
        transport=ASGITransport(
            app=create_sandbox_app(data_dir=cfg.storage.data_dir, token=SANDBOX_TOKEN, uploads=cfg.uploads)
        ),
    )
    state = AppState(
        config=cfg,
        engine=engine,
        session_factory=factory,
        runs=RunRepository(factory),
        commands=RunCommandRepository(factory),
        usage=RunUsageRepository(factory),
        decisions=BudgetDecisionRepository(factory),
        sessions=AdminSessionRepository(factory),
        reports=ReportRepository(factory),
        artifacts=ArtifactRepository(factory),
        uploads=UploadRepository(factory),
        packages=GraderPackageRepository(factory),
        sandbox=sandbox_client,
        audit=AuditRepository(factory),
        admin_password_hash=hash_password(password),
        login_limiter=LoginRateLimiter(),
        index_html=index_html,
    )
    return create_app(state), state
