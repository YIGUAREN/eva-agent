from pathlib import Path

import pytest

from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind, PastedInput, RunInput
from eva_agent.runtime.manager import RunManager
from tests.support.target_runtime import runtime_config


def _paste_run_input() -> RunInput:
    return RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q", response="a"))


async def test_manager_persists_audit_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "x")
    config = runtime_config(tmp_path)
    async with RunManager.open(config) as manager:
        await manager._require_repo().create(
            "run-1",
            _paste_run_input(),
            Budget(max_cost_usd=1, max_wall_clock_s=10, max_target_requests=1),
        )
        assert manager._audit is not None
        await manager._audit.record(
            AuditEvent(run_id="run-1", actor="cli:admin", action="active_scan.requested", outcome="allowed")
        )
        rows = await manager.audit_events("run-1")
    assert [row.action for row in rows] == ["active_scan.requested"]


async def test_manager_dependencies_share_the_audit_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "x")
    config = runtime_config(tmp_path)
    async with RunManager.open(config) as manager:
        await manager._require_repo().create(
            "run-1",
            _paste_run_input(),
            Budget(max_cost_usd=1, max_wall_clock_s=10, max_target_requests=1),
        )
        # The graph-facing sink must be the same repository that audit_events reads,
        # otherwise graph/tool audit events are written to the noop sink and lost.
        assert manager._require_dependencies().audit is manager._audit
        await manager._require_dependencies().audit.record(
            AuditEvent(run_id="run-1", actor="cli:admin", action="active_scan.started", outcome="allowed")
        )
        rows = await manager.audit_events("run-1")
    assert [row.action for row in rows] == ["active_scan.started"]


async def test_audit_events_requires_setup(tmp_path: Path) -> None:
    manager = RunManager(runtime_config(tmp_path))
    with pytest.raises(RuntimeError, match="setup"):
        await manager.audit_events("run-1")
