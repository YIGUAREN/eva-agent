"""Audit sink protocol and defensive payload sanitization."""

from __future__ import annotations

from typing import Any, Protocol

from eva_agent.domain.audit import AuditEvent

REDACTED = "<redacted>"
_SENSITIVE = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "authorization",
        "password",
        "secret",
        "token",
    }
)
MAX_STRING = 512
MAX_ITEMS = 32
MAX_DEPTH = 3


def sanitize_detail(detail: dict[str, Any]) -> dict[str, Any]:
    return _sanitize(detail, depth=0)


def _sanitize(value: Any, *, depth: int) -> Any:
    if depth >= MAX_DEPTH:
        return "<truncated>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:MAX_ITEMS]:
            name = str(key)[:128]
            normalized = name.lower().replace("-", "_")
            result[name] = REDACTED if normalized in _SENSITIVE else _sanitize(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth=depth + 1) for item in list(value)[:MAX_ITEMS]]
    if isinstance(value, str):
        return value[:MAX_STRING]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


class AuditSink(Protocol):
    async def record(self, event: AuditEvent) -> None: ...


class NoopAuditSink:
    async def record(self, event: AuditEvent) -> None:
        return None


noop_audit_sink = NoopAuditSink()
