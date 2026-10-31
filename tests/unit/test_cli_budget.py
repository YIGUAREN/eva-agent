import argparse
from contextlib import asynccontextmanager

from eva_agent.cli import _status
from eva_agent.domain.run import RunStatus
from eva_agent.runtime.manager import RunManager, RunStatusView
from tests.support.runtime import fake_config


class FakeManager:
    async def status(self, run_id: str) -> RunStatusView:
        return RunStatusView(run_id=run_id, status=RunStatus.COMPLETED, plan=None, report_json_path=None, errors=[])


async def test_cli_status_opens_an_unlimited_ledger(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    @asynccontextmanager
    async def fake_open(config, **kwargs):
        seen.update(kwargs)
        yield FakeManager()

    monkeypatch.setattr(RunManager, "open", fake_open)
    assert await _status(argparse.Namespace(run_id="run-1"), fake_config(tmp_path)) == 0
    assert seen["enforce_budget"] is False
