from decimal import Decimal
from pathlib import Path

import pytest
from httpx import AsyncClient

from eva_agent.budget.types import RunUsage
from eva_agent.config.settings import AppConfig, PricingSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import RunStatus
from eva_agent.runtime.budget_decisions import BudgetOverrunRequired, InvalidBudgetExtension
from eva_agent.runtime.manager import RunManager, RunStateError
from eva_agent.server.state import AppState
from eva_agent.storage.budget import BudgetDecisionRepository
from eva_agent.storage.commands import RunCommandRepository
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.repository import RunRepository
from eva_agent.worker.runner import RunWorker
from tests.support.api import PASTE_PAYLOAD, csrf_headers, login
from tests.support.runtime import budget, dependencies, fake_config, paste_input

GATE_BUDGET = Budget(max_cost_usd=0.0001, max_wall_clock_s=60.0, max_target_requests=90)
EXTENDED_BUDGET = Budget(max_cost_usd=2.0, max_wall_clock_s=60.0, max_target_requests=90)


async def _finish_pending(state: AppState, run_id: str) -> None:
    """Drain the start command the create route enqueues, as the worker would."""
    for command in await state.commands.list_for_run(run_id):
        if command.status == "pending":
            await state.commands.finish(command.id, "done")


async def _waiting_run(client: AsyncClient, state: AppState) -> tuple[str, str]:
    csrf = await login(client)
    created = await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))
    run_id = created.json()["run_id"]
    await _finish_pending(state, run_id)
    await state.runs.set_status(run_id, RunStatus.WAITING_BUDGET)
    await state.usage.save(
        run_id,
        RunUsage(
            spent_usd=Decimal("0.9"),
            reserved_usd=Decimal("0"),
            spent_wall_clock_s=10.0,
            target_requests=3,
        ),
    )
    await state.decisions.record(
        run_id,
        1,
        "gate",
        {
            "finalize_estimate": 0.2,
            "remaining": 0.1,
            "spent": 0.9,
            "truncated_dimensions": ["quality"],
        },
    )
    return run_id, csrf


async def test_summarize_without_confirmation_returns_the_overrun_details(
    client: AsyncClient, state: AppState
) -> None:
    run_id, csrf = await _waiting_run(client, state)
    response = await client.post(
        f"/api/runs/{run_id}/budget/decision",
        json={"action": "summarize"},
        headers=csrf_headers(csrf),
    )
    assert response.status_code == 409
    body = response.json()
    assert body["error"]["code"] == "budget_overrun"
    assert body["error"]["details"] == {"finalize_estimate": 0.2, "remaining": 0.1}


async def test_summarize_with_confirmation_enqueues_a_command(
    client: AsyncClient, state: AppState
) -> None:
    run_id, csrf = await _waiting_run(client, state)
    response = await client.post(
        f"/api/runs/{run_id}/budget/decision",
        json={"action": "summarize", "confirm_overrun": True},
        headers=csrf_headers(csrf),
    )
    assert response.status_code == 202
    commands = await state.commands.list_for_run(run_id)
    assert commands[-1].kind == "budget_decision"
    assert commands[-1].payload_json == {"action": "summarize", "confirm_overrun": True}


async def test_extend_rejects_a_limit_not_above_spent(client: AsyncClient, state: AppState) -> None:
    run_id, csrf = await _waiting_run(client, state)
    response = await client.post(
        f"/api/runs/{run_id}/budget/decision",
        json={
            "action": "extend",
            "budget": {"max_cost_usd": 0.9, "max_wall_clock_s": 60.0, "max_target_requests": 90},
        },
        headers=csrf_headers(csrf),
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["details"] == {"field": "max_cost_usd"}


async def test_extend_with_all_limits_above_spent_enqueues_the_budget(
    client: AsyncClient, state: AppState
) -> None:
    run_id, csrf = await _waiting_run(client, state)
    payload = {
        "action": "extend",
        "budget": {"max_cost_usd": 2.0, "max_wall_clock_s": 60.0, "max_target_requests": 90},
    }
    response = await client.post(
        f"/api/runs/{run_id}/budget/decision",
        json=payload,
        headers=csrf_headers(csrf),
    )
    assert response.status_code == 202
    commands = await state.commands.list_for_run(run_id)
    assert commands[-1].payload_json == payload


async def test_duplicate_decision_after_resume_returns_the_current_state(
    client: AsyncClient, state: AppState
) -> None:
    run_id, csrf = await _waiting_run(client, state)
    await state.runs.set_status(run_id, RunStatus.RUNNING)
    before = [(command.id, command.kind) for command in await state.commands.list_for_run(run_id)]

    response = await client.post(
        f"/api/runs/{run_id}/budget/decision",
        json={"action": "summarize", "confirm_overrun": True},
        headers=csrf_headers(csrf),
    )

    assert response.status_code == 202
    assert response.json()["queued"] == ""
    after = [(command.id, command.kind) for command in await state.commands.list_for_run(run_id)]
    assert after == before


async def test_decide_claims_the_gate_row_exactly_once(tmp_path: Path) -> None:
    config = fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    factory = create_session_factory(engine)
    try:
        await RunRepository(factory).create("run-decide-idem", paste_input(), budget())
        decisions = BudgetDecisionRepository(factory)
        gate = {"finalize_estimate": 0.2, "remaining": 0.1}
        assert await decisions.record("run-decide-idem", 1, "gate", gate) is True

        assert await decisions.decide(
            "run-decide-idem", 1, "summarize", {**gate, "decision": {"action": "summarize"}}
        ) is True
        # The second delivery for the same seq is a no-op, even with another payload.
        assert await decisions.decide("run-decide-idem", 1, "extend", {"decision": {"action": "extend"}}) is False
        row = await decisions.get("run-decide-idem", 1)
        assert row is not None and row.action == "summarize"
    finally:
        await engine.dispose()


def _priced_config(tmp_path: Path) -> AppConfig:
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "models:\n"
        "  - provider: openai-compatible\n"
        "    model: fake\n"
        "    input_per_million: 1000.0\n"
        "    output_per_million: 1000.0\n",
        encoding="utf-8",
    )
    return fake_config(tmp_path).model_copy(update={"pricing": PricingSettings(path=str(pricing))})


async def _gate_run(manager: RunManager, run_id: str) -> str:
    await manager.start(paste_input(), GATE_BUDGET, run_id=run_id)
    view = await manager.approve(run_id, approved=True)
    assert view.status is RunStatus.WAITING_BUDGET
    return run_id


async def test_sync_reports_waiting_budget_and_generic_resume_is_refused(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-sync")

        assert (await manager.status(run_id)).status is RunStatus.WAITING_BUDGET
        with pytest.raises(RunStateError):
            await manager.resume(run_id)


async def test_decide_budget_requires_confirmation_then_resumes_once(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-confirm")

        with pytest.raises(BudgetOverrunRequired):
            await manager.decide_budget(run_id, action="summarize")
        assert (await manager.status(run_id)).status is RunStatus.WAITING_BUDGET

        view = await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)
        assert view.status is RunStatus.COMPLETED

        duplicate = await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)
        assert duplicate.status is RunStatus.COMPLETED


async def test_decide_budget_rejects_an_extension_not_above_spent(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-invalid")
        assert manager._usage_repo is not None
        await manager._usage_repo.save(run_id, RunUsage(spent_usd=Decimal("0.5"), spent_wall_clock_s=10.0))

        with pytest.raises(InvalidBudgetExtension) as exc:
            await manager.decide_budget(
                run_id,
                action="extend",
                budget=Budget(max_cost_usd=0.5, max_wall_clock_s=60.0, max_target_requests=90),
            )
        assert exc.value.field == "max_cost_usd"
        assert (await manager.status(run_id)).status is RunStatus.WAITING_BUDGET


async def test_extend_raises_the_live_ledger_limit(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-extend")

        view = await manager.decide_budget(run_id, action="extend", budget=EXTENDED_BUDGET)
        assert view.status is RunStatus.COMPLETED

        ledger = await manager._ledger_for(run_id)
        await ledger.reserve(Decimal("1.0"), kind="probe")
        assert ledger.snapshot().remaining_usd >= Decimal("0.9")
        row = await manager._require_repo().get(run_id)
        assert row is not None
        assert row.budget_json["max_cost_usd"] == 2.0


class _ProcessDeath(BaseException):
    """Simulate the process dying after the decision row was written."""


class _DyingGraph:
    def __init__(self, inner: object) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    async def ainvoke(self, *args: object, **kwargs: object) -> object:
        raise _ProcessDeath


async def test_crash_after_decide_recovers_the_recorded_decision_on_retry(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-crash")
        real_graph = manager._graph
        manager._graph = _DyingGraph(real_graph)
        try:
            with pytest.raises(_ProcessDeath):
                await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)
        finally:
            manager._graph = real_graph

        # The claim and the decision were persisted, but the graph never resumed.
        assert manager._decisions is not None
        assert await manager._decisions.latest_gate(run_id) is None
        assert (await manager.status(run_id)).status is RunStatus.RUNNING

        view = await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)
        assert view.status is RunStatus.COMPLETED


async def test_worker_budget_decision_command_resumes_the_run(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-worker")
        commands = RunCommandRepository(manager.session_factory)
        await commands.enqueue(
            run_id,
            "budget_decision",
            {"action": "summarize", "confirm_overrun": True},
        )
        worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir, poll_interval_s=0.01)

        assert await worker.tick() is True
        assert (await manager.status(run_id)).status is RunStatus.COMPLETED
        rows = await commands.list_for_run(run_id)
        assert rows[-1].kind == "budget_decision"
        assert rows[-1].status == "done"
