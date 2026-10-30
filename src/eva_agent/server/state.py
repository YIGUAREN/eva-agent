"""Shared application state for the web process."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from eva_agent.config.settings import AppConfig
from eva_agent.server.ratelimit import LoginRateLimiter
from eva_agent.storage.audit import AuditRepository
from eva_agent.storage.commands import RunCommandRepository
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
from eva_agent.storage.repository import RunRepository
from eva_agent.storage.sessions import AdminSessionRepository


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class AppState:
    config: AppConfig
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    runs: RunRepository
    commands: RunCommandRepository
    sessions: AdminSessionRepository
    reports: ReportRepository
    artifacts: ArtifactRepository
    audit: AuditRepository
    admin_password_hash: str
    login_limiter: LoginRateLimiter
    index_html: Path
    now: Callable[[], datetime] = _utcnow
