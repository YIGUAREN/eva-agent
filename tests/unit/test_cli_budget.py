import argparse
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import pytest

from eva_agent.cli import _approve, _report, _start, _status
from eva_agent.config.settings import AppConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import RunInput, RunStatus
from eva_agent.runtime.manager import RunManager, RunStatusView
from tests.support.runtime import fake_config

Handler = Callable[[argparse.Namespace, AppConfig], Awaitable[int]]


def _view(run_id: str) -> RunStatusView:
    return RunStatusView(run_id=run_id, status=RunStatus.COMPLETED, plan=None, report_json_path=None, errors=[])


class FakeManager:
    async def start(self, run_input: RunInput, budget: Budget) -> str:
        return "run-1"

    async def approve(self, run_id: str, *, approved: bool, plan: dict | None) -> RunStatusView:
        return _view(run_id)

    async def status(self, run_id: str) -> RunStatusView:
        return _view(run_id)

    async def report(self, run_id: str) -> dict[str, object]:
        return {"run_id": run_id}


def _start_args() -> argparse.Namespace:
    return argparse.Namespace(
        target=None,
        prompt=[],
        dataset=None,
        query="q",
        response="a",
        context=None,
        objective=None,
        dataset_format="csv",
        column_map=[],
        max_cost_usd=1.0,
        max_wall_clock_s=60.0,
        max_target_requests=1,
    )


HANDLERS: list[tuple[str, Handler, argparse.Namespace]] = [
    ("start", _start, _start_args()),
    ("approve", _approve, argparse.Namespace(run_id="run-1", approve=True, reject=False, plan_file=None)),
    ("status", _status, argparse.Namespace(run_id="run-1")),
    ("report", _report, argparse.Namespace(run_id="run-1")),
]


@pytest.mark.parametrize(("name", "handler", "args"), HANDLERS, ids=[item[0] for item in HANDLERS])
async def test_cli_opens_an_unlimited_ledger(
    name: str, handler: Handler, args: argparse.Namespace, tmp_path, monkeypatch
) -> None:
    seen: dict[str, object] = {}

    @asynccontextmanager
    async def fake_open(config, **kwargs):
        seen.update(kwargs)
        yield FakeManager()

    monkeypatch.setattr(RunManager, "open", fake_open)
    assert await handler(args, fake_config(tmp_path)) == 0
    assert seen["enforce_budget"] is False
