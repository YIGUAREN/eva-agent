import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient
from langgraph.types import Command
from sqlalchemy import func, select

from eva_agent.budget.types import RunUsage
from eva_agent.config.settings import AppConfig, PricingSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import RunStatus
from eva_agent.pricing import PricingTable
from eva_agent.runtime.budget_decisions import BudgetOverrunRequired, InvalidBudgetExtension
from eva_agent.runtime.manager import RunManager, RunStateError
from eva_agent.server.state import AppState
from eva_agent.storage.budget import BudgetDecisionRepository
from eva_agent.storage.commands import RunCommandRepository
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import ArtifactRow, ReportRow
from eva_agent.storage.repository import RunRepository
from eva_agent.worker.runner import RunWorker
from tests.support.api import PASTE_PAYLOAD, csrf_headers, login
from tests.support.runtime import budget, dependencies, fake_config, paste_input

# Grading now reserves from the ledger, so the gate budget must cover the fake
# run's two grader specs (~1.204 USD at the priced-config rates) while staying
# below the finalize estimate (~1.33 USD) that triggers the pause.  The margin
# above grading is thin: a tokenizer or sample-plan change can flip the assertion.
GATE_BUDGET = Budget(max_cost_usd=1.25, max_wall_clock_s=60.0, max_target_requests=90)
EXTENDED_BUDGET = Budget(max_cost_usd=3.5, max_wall_clock_s=60.0, max_target_requests=90)


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
    await state.decisions.decide(
        run_id, 1, "summarize", {"decision": {"action": "summarize", "confirm_overrun": True}}
    )
    await state.runs.set_status(run_id, RunStatus.COMPLETED)
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


async def test_decision_while_queued_is_rejected(client: AsyncClient, state: AppState) -> None:
    run_id, csrf = await _waiting_run(client, state)
    await state.runs.set_status(run_id, RunStatus.QUEUED)

    response = await client.post(
        f"/api/runs/{run_id}/budget/decision",
        json={"action": "summarize", "confirm_overrun": True},
        headers=csrf_headers(csrf),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_state"


async def test_route_enqueues_a_decision_for_a_wedged_run(
    client: AsyncClient, state: AppState
) -> None:
    run_id, csrf = await _waiting_run(client, state)
    # The gate checkpoint was written but the manager never synced the row.
    await state.runs.set_status(run_id, RunStatus.RUNNING)

    response = await client.post(
        f"/api/runs/{run_id}/budget/decision",
        json={"action": "summarize", "confirm_overrun": True},
        headers=csrf_headers(csrf),
    )

    assert response.status_code == 202
    assert response.json()["queued"] == "budget_decision"


async def test_cancel_while_waiting_budget_enqueues_the_cancel(
    client: AsyncClient, state: AppState
) -> None:
    run_id, csrf = await _waiting_run(client, state)

    response = await client.post(f"/api/runs/{run_id}/cancel", headers=csrf_headers(csrf))

    assert response.status_code == 202
    commands = await state.commands.list_for_run(run_id)
    assert commands[-1].kind == "cancel"


async def test_route_enqueues_a_recovery_for_a_resumed_but_unsynced_run(
    client: AsyncClient, state: AppState
) -> None:
    run_id, csrf = await _waiting_run(client, state)
    # The decision resumed the graph, but _sync never persisted the result: the
    # row stays RUNNING and no pending gate remains.  The duplicate submission
    # must not silently succeed; it enqueues the recorded decision so the worker
    # converges the row and runs post-processing.
    decision = {"action": "summarize", "confirm_overrun": True}
    await state.decisions.decide(run_id, 1, "summarize", {"decision": decision})
    await state.runs.set_status(run_id, RunStatus.RUNNING)

    response = await client.post(
        f"/api/runs/{run_id}/budget/decision",
        json={"action": "summarize", "confirm_overrun": True},
        headers=csrf_headers(csrf),
    )

    assert response.status_code == 202
    assert response.json()["queued"] == "budget_decision"
    commands = await state.commands.list_for_run(run_id)
    assert commands[-1].kind == "budget_decision"
    assert commands[-1].payload_json == decision


async def test_recovery_audit_records_the_replayed_decision(
    client: AsyncClient, state: AppState
) -> None:
    """Finding 2: the audit row must not claim a different action was accepted."""
    run_id, csrf = await _waiting_run(client, state)
    decision = {"action": "summarize", "confirm_overrun": True}
    await state.decisions.decide(run_id, 1, "summarize", {"decision": decision})
    await state.runs.set_status(run_id, RunStatus.RUNNING)

    response = await client.post(
        f"/api/runs/{run_id}/budget/decision",
        json={
            "action": "extend",
            "budget": {"max_cost_usd": 2.0, "max_wall_clock_s": 60.0, "max_target_requests": 90},
        },
        headers=csrf_headers(csrf),
    )

    assert response.status_code == 202
    commands = await state.commands.list_for_run(run_id)
    assert commands[-1].payload_json == decision
    events = await state.audit.list_for_run(run_id)
    assert events[-1].action == "run.budget.summarize"
    assert events[-1].outcome == "replayed"


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
        assert row.budget_json["max_cost_usd"] == 3.5


class _ProcessDeath(BaseException):
    """Simulate the process dying after the decision row was written."""


class _DyingGraph:
    def __init__(self, inner: object) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    async def ainvoke(self, *args: object, **kwargs: object) -> object:
        raise _ProcessDeath


class _CountingGraph:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    async def ainvoke(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return await self._inner.ainvoke(*args, **kwargs)


async def _count_rows(factory: Any, model: Any, run_id: str) -> int:
    async with factory() as session:
        count = await session.scalar(
            select(func.count()).select_from(model).where(model.run_id == run_id)
        )
    return int(count or 0)


async def _advance_one_superstep(manager: RunManager, run_id: str, decision: dict[str, Any]) -> None:
    """Drive the resumed graph for one superstep, then abandon it like a crash."""
    graph = manager._require_graph()
    async with manager._budget_scope(run_id):
        stream = graph.astream(Command(resume=decision), manager._thread(run_id), stream_mode="updates")
        try:
            async for _chunk in stream:
                break
        finally:
            await stream.aclose()


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


async def test_concurrent_decisions_replay_the_crash_window_once(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-concurrent")
        real_graph = manager._graph
        manager._graph = _DyingGraph(real_graph)
        try:
            with pytest.raises(_ProcessDeath):
                await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)
        finally:
            manager._graph = real_graph

        counting = _CountingGraph(real_graph)
        manager._graph = counting
        try:
            views = await asyncio.gather(
                manager.decide_budget(run_id, action="summarize", confirm_overrun=True),
                manager.decide_budget(run_id, action="summarize", confirm_overrun=True),
            )
        finally:
            manager._graph = real_graph

        assert counting.calls == 1
        assert {view.status for view in views} == {RunStatus.COMPLETED}


async def test_concurrent_decisions_on_a_parked_run_resume_once(tmp_path: Path) -> None:
    """Minor 4: two native duplicate submissions resume a normally parked run once."""
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-native-concurrent")
        real_graph = manager._graph
        counting = _CountingGraph(real_graph)
        manager._graph = counting
        try:
            views = await asyncio.gather(
                manager.decide_budget(run_id, action="summarize", confirm_overrun=True),
                manager.decide_budget(run_id, action="summarize", confirm_overrun=True),
            )
        finally:
            manager._graph = real_graph

        assert counting.calls == 1
        assert {view.status for view in views} == {RunStatus.COMPLETED}


async def test_resume_recorded_decision_replays_only_a_running_row(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-direct-recovery")
        real_graph = manager._graph
        manager._graph = _DyingGraph(real_graph)
        try:
            with pytest.raises(_ProcessDeath):
                await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)
        finally:
            manager._graph = real_graph

        counting = _CountingGraph(real_graph)
        manager._graph = counting
        try:
            first = await manager._resume_recorded_decision(run_id)
            second = await manager._resume_recorded_decision(run_id)
        finally:
            manager._graph = real_graph

        assert first.status is RunStatus.COMPLETED
        assert second.status is RunStatus.COMPLETED
        assert counting.calls == 1


async def test_decision_recovers_a_run_wedged_before_sync(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-wedged")
        repo = manager._require_repo()
        # The gate checkpoint landed but the row was never synced to WAITING_BUDGET.
        assert await repo.claim(
            run_id,
            from_statuses=(RunStatus.WAITING_BUDGET,),
            to_status=RunStatus.RUNNING,
        )

        view = await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)

        assert view.status is RunStatus.COMPLETED


async def test_sync_failure_after_the_gate_pause_leaves_the_run_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = "run-gate-sync-failure"
        await manager.start(paste_input(), GATE_BUDGET, run_id=run_id)
        repo = manager._require_repo()

        async def boom(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        with monkeypatch.context() as patch:
            patch.setattr(repo, "set_plan", boom)
            with pytest.raises(OSError):
                await manager.approve(run_id, approved=True)

        assert (await manager.status(run_id)).status is RunStatus.WAITING_BUDGET
        view = await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)
        assert view.status is RunStatus.COMPLETED


async def test_resume_finished_before_sync_converges_to_completed_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Important 1: the resume reached END but _sync failed, leaving the row RUNNING."""
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-finished-before-sync-fail")
        repo = manager._require_repo()

        async def boom(*args: object, **kwargs: object) -> None:
            raise OSError("database is locked")

        with monkeypatch.context() as patch:
            patch.setattr(repo, "set_plan", boom)
            with pytest.raises(OSError):
                await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)

        # The graph is at END, the decision is recorded, but the row was never synced.
        assert (await manager.status(run_id)).status is RunStatus.RUNNING
        assert not await manager.paused_at_budget_gate(run_id)

        view = await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)

        assert view.status is RunStatus.COMPLETED


async def test_worker_recovers_a_run_that_finished_before_sync(tmp_path: Path) -> None:
    """Important 1: a worker retry must converge and run post-processing exactly once."""
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-worker-finished-before-sync")
        repo = manager._require_repo()
        decisions = manager._decisions
        assert decisions is not None
        gate = await decisions.latest_gate(run_id)
        assert gate is not None
        decision = {"action": "summarize", "confirm_overrun": True}
        assert await repo.claim(
            run_id,
            from_statuses=(RunStatus.WAITING_BUDGET,),
            to_status=RunStatus.RUNNING,
        )
        assert await decisions.decide(
            run_id, int(gate.seq), "summarize", {**dict(gate.payload_json), "decision": decision}
        )
        # The process dies after the graph reaches END but before _sync persists it.
        await manager._invoke_resume(run_id, decision)
        assert (await manager.status(run_id)).status is RunStatus.RUNNING
        assert not await manager.paused_at_budget_gate(run_id)

        commands = RunCommandRepository(manager.session_factory)
        await commands.enqueue(run_id, "budget_decision", decision)
        worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir, poll_interval_s=0.01)

        assert await worker.tick() is True

        assert (await manager.status(run_id)).status is RunStatus.COMPLETED
        assert await worker.reports.get(run_id) is not None
        assert await worker.artifacts.get(run_id, "report.json") is not None
        reports_before = await _count_rows(manager.session_factory, ReportRow, run_id)
        artifacts_before = await _count_rows(manager.session_factory, ArtifactRow, run_id)
        assert reports_before == 1
        assert artifacts_before >= 1

        # Replaying the recovery must not duplicate report/artifact rows, and the
        # replay itself must succeed: a swallowed duplicate-write error would
        # leave the command row `failed` while row counts stayed unchanged.
        replay_command = await commands.enqueue(run_id, "budget_decision", decision)
        assert await worker.tick() is True
        replay = await commands.get(replay_command)
        assert replay is not None
        assert replay.status == "done"
        assert replay.error is None
        assert await worker.artifacts.get(run_id, "report.json") is not None
        assert await _count_rows(manager.session_factory, ReportRow, run_id) == reports_before
        assert await _count_rows(manager.session_factory, ArtifactRow, run_id) == artifacts_before


async def test_mid_graph_crash_after_the_decision_recovers_and_completes(tmp_path: Path) -> None:
    """Finding 1: the decision resumed the graph, which then died mid-flight."""
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-mid-graph")
        repo = manager._require_repo()
        decisions = manager._decisions
        assert decisions is not None
        gate = await decisions.latest_gate(run_id)
        assert gate is not None
        decision = {"action": "summarize", "confirm_overrun": True}
        assert await repo.claim(
            run_id,
            from_statuses=(RunStatus.WAITING_BUDGET,),
            to_status=RunStatus.RUNNING,
        )
        assert await decisions.decide(
            run_id, int(gate.seq), "summarize", {**dict(gate.payload_json), "decision": decision}
        )
        # The decision resumed the graph, one tail superstep ran, and the
        # process died before the invocation could return (merge/analyze/report
        # still queued).
        await _advance_one_superstep(manager, run_id, decision)
        snapshot = await manager._require_graph().aget_state(manager._thread(run_id))
        next_nodes = set(snapshot.next or ())
        assert next_nodes
        assert "budget_gate" not in next_nodes
        assert "approve" not in next_nodes
        assert (await manager.status(run_id)).status is RunStatus.RUNNING
        assert await decisions.latest_gate(run_id) is None

        commands = RunCommandRepository(manager.session_factory)
        worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir, poll_interval_s=0.01)
        real_graph = manager._graph
        counting = _CountingGraph(real_graph)
        manager._graph = counting
        try:
            await commands.enqueue(run_id, "budget_decision", decision)
            assert await worker.tick() is True
        finally:
            manager._graph = real_graph

        assert counting.calls == 1
        assert (await manager.status(run_id)).status is RunStatus.COMPLETED
        assert await worker.reports.get(run_id) is not None
        assert await worker.artifacts.get(run_id, "report.json") is not None
        reports_before = await _count_rows(manager.session_factory, ReportRow, run_id)
        artifacts_before = await _count_rows(manager.session_factory, ArtifactRow, run_id)
        assert reports_before == 1

        # A second recovery command must neither invoke the graph again nor
        # write the report/artifacts a second time.
        extra = _CountingGraph(real_graph)
        manager._graph = extra
        try:
            extra_command = await commands.enqueue(run_id, "budget_decision", decision)
            assert await worker.tick() is True
        finally:
            manager._graph = real_graph

        replay = await commands.get(extra_command)
        assert replay is not None
        assert replay.status == "done"
        assert replay.error is None
        assert extra.calls == 0
        assert await _count_rows(manager.session_factory, ReportRow, run_id) == reports_before
        assert await _count_rows(manager.session_factory, ArtifactRow, run_id) == artifacts_before


async def test_worker_resume_keeps_a_decided_gate_replayable(tmp_path: Path) -> None:
    """Minor 2: reconciling must not park a row whose decision is already recorded."""
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-resume-replayable")
        real_graph = manager._graph
        manager._graph = _DyingGraph(real_graph)
        try:
            with pytest.raises(_ProcessDeath):
                await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)
        finally:
            manager._graph = real_graph

        # RUNNING, decision recorded, checkpoint still parked at the gate.
        assert (await manager.status(run_id)).status is RunStatus.RUNNING
        assert await manager.paused_at_budget_gate(run_id)

        commands = RunCommandRepository(manager.session_factory)
        await commands.enqueue(run_id, "resume")
        worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir, poll_interval_s=0.01)
        assert await worker.tick() is True

        # Only a RUNNING row can replay, so the reconcile must leave it RUNNING.
        assert (await manager.status(run_id)).status is RunStatus.RUNNING

        counting = _CountingGraph(real_graph)
        manager._graph = counting
        try:
            await commands.enqueue(
                run_id,
                "budget_decision",
                {"action": "summarize", "confirm_overrun": True},
            )
            assert await worker.tick() is True
        finally:
            manager._graph = real_graph

        assert counting.calls == 1
        assert (await manager.status(run_id)).status is RunStatus.COMPLETED


async def test_lost_decision_claim_remains_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Minor 5: `decide` returning False must not leave the row claimed RUNNING."""
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-lost-decision")
        decisions = manager._decisions
        assert decisions is not None

        async def lost(*args: object, **kwargs: object) -> bool:
            return False

        with monkeypatch.context() as patch:
            patch.setattr(decisions, "decide", lost)
            view = await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)

        assert view.status is RunStatus.WAITING_BUDGET

        # A later retry can still decide and finish the run.
        final = await manager.decide_budget(run_id, action="summarize", confirm_overrun=True)
        assert final.status is RunStatus.COMPLETED


async def test_worker_reconciles_a_wedged_run_instead_of_failing_it(tmp_path: Path) -> None:
    config = _priced_config(tmp_path)
    async with RunManager.open(config, dependencies=dependencies(tmp_path), enforce_budget=True) as manager:
        run_id = await _gate_run(manager, "run-gate-worker-recover")
        repo = manager._require_repo()
        assert await repo.claim(
            run_id,
            from_statuses=(RunStatus.WAITING_BUDGET,),
            to_status=RunStatus.RUNNING,
        )
        commands = RunCommandRepository(manager.session_factory)
        await commands.enqueue(run_id, "start")
        worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir, poll_interval_s=0.01)

        assert await worker.tick() is True

        assert (await manager.status(run_id)).status is RunStatus.WAITING_BUDGET
        rows = await commands.list_for_run(run_id)
        assert rows[-1].status == "done"


async def test_unenforced_manager_does_not_load_the_pricing_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[Path] = []

    def record(path: Path) -> PricingTable:
        loads.append(path)
        raise AssertionError("pricing must not load without budget enforcement")

    monkeypatch.setattr(PricingTable, "load", staticmethod(record))
    config = fake_config(tmp_path).model_copy(
        update={"pricing": PricingSettings(path=str(tmp_path / "missing-pricing.yaml"))}
    )
    async with RunManager.open(config, dependencies=dependencies(tmp_path)):
        pass
    assert loads == []


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
