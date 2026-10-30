from pathlib import Path

import pytest

from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind, PastedInput, RunInput
from eva_agent.runtime.manager import RunManager
from tests.support.target_runtime import runtime_config


async def test_manager_persists_audit_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "x")
    config = runtime_config(tmp_path)
    async with RunManager.open(config) as manager:
        await manager._require_repo().create(
            "run-1",
            RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q", response="a")),
            Budget(max_cost_usd=1, max_wall_clock_s=10, max_target_requests=1),
        )
        assert manager._audit is not None
        await manager._audit.record(
            AuditEvent(run_id="run-1", actor="cli:admin", action="active_scan.requested", outcome="allowed")
        )
        rows = await manager.audit_events("run-1")
    assert [row.action for row in rows] == ["active_scan.requested"]
