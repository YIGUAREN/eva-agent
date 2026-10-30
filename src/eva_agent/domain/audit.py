"""Audit events for privileged operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    run_id: str
    actor: str
    action: str
    outcome: str
    target: str | None = None
    authorization_ref: str | None = None
    scan_kind: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
