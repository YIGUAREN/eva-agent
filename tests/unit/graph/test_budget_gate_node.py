"""Unit tests for the prepare half of the split budget gate."""

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.build import _route_after_budget_gate_prepare
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.nodes.budget_gate import budget_gate_prepare
from eva_agent.graph.state import EvalState
from tests.support.runtime import dependencies

RUN_ID = "run-1"
BUDGET = Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5)


class NoopStore:
    async def load(self, run_id: str) -> None:
        return None

    async def save(self, run_id: str, usage: Any) -> None:
        return None


def _models() -> BudgetModels:
    model = ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")
    return BudgetModels(agent=model, judge=model, attack=model)


class RecordingGateStore:
    def __init__(self) -> None:
        self.rows: dict[int, dict[str, Any]] = {}
        self.record_calls: list[tuple[int, str, dict[str, Any]]] = []
        self.pending: SimpleNamespace | None = None

    async def next_seq(self, run_id: str) -> int:
        return max(self.rows, default=0) + 1

    async def record(self, run_id: str, seq: int, action: str, payload: dict[str, Any]) -> bool:
        self.record_calls.append((seq, action, payload))
        self.rows[seq] = payload
        return True

    async def latest_gate(self, run_id: str) -> SimpleNamespace | None:
        return self.pending


class DuplicateGateStore(RecordingGateStore):
    async def next_seq(self, run_id: str) -> int:
        return 1

    async def record(self, run_id: str, seq: int, action: str, payload: dict[str, Any]) -> bool:
        self.record_calls.append((seq, action, payload))
        return False


def _deps(
    tmp_path,
    *,
    store: Any,
    events: list[tuple[str, dict]],
    estimator: Any,
) -> GraphDependencies:
    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    return replace(
        dependencies(tmp_path, event_sink=sink),
        finalize_estimator=estimator,
        gate_store=store,
    )


def _ledger(spent: str = "0", *, unlimited: bool = False) -> BudgetLedger:
    ledger = (
        BudgetLedger.unlimited(BUDGET)
        if unlimited
        else BudgetLedger(BUDGET, store=NoopStore())
    )
    if spent != "0":
        ledger._reserve_sync(Decimal(spent))
        ledger._settle_sync(Decimal(spent), Decimal(spent))
    return ledger


def _state(results: list[DimensionResult]) -> EvalState:
    return cast(
        EvalState,
        {
            "run_id": RUN_ID,
            "budget": BUDGET,
            "expert_results": results,
            "status": RunStatus.RUNNING,
        },
    )


def _result(
    *,
    reason: str | None = "budget",
    dimension: Dimension = Dimension.QUALITY,
) -> DimensionResult:
    return DimensionResult(
        dimension=dimension,
        status="partial",
        score=None,
        confidence=0.0,
        summary="truncated",
        usage=Usage(),
        truncated_reason=reason,
    )


def _constant_estimator(value: str = "0.5"):
    return lambda state: Decimal(value)


def _failing_estimator(state):
    raise AssertionError("the estimator must not run on a bypass")


async def test_prepare_bypasses_before_the_estimator_when_no_context_is_bound(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = RecordingGateStore()
    deps = _deps(tmp_path, store=store, events=events, estimator=_failing_estimator)

    result = await budget_gate_prepare(_state([_result()]), deps)

    assert result == {
        "status": RunStatus.RUNNING,
        "finalize_estimate_usd": 0.0,
        "finalize_overrun": False,
        "budget_gate": None,
        "budget_gate_seq": None,
    }
    assert events == []
    assert store.record_calls == []


async def test_prepare_bypasses_before_the_estimator_for_an_unlimited_ledger(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = RecordingGateStore()
    deps = _deps(tmp_path, store=store, events=events, estimator=_failing_estimator)

    with bind_budget(_ledger(unlimited=True), CostEstimator(None), _models()):
        result = await budget_gate_prepare(_state([_result()]), deps)

    assert result["budget_gate"] is None
    assert result["finalize_estimate_usd"] == 0.0
    assert events == []
    assert store.record_calls == []


async def test_prepare_fails_closed_when_an_enforced_ledger_has_no_estimator(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    deps = _deps(tmp_path, store=RecordingGateStore(), events=events, estimator=None)

    with (
        bind_budget(_ledger("0.95"), CostEstimator(None), _models()),
        pytest.raises(RuntimeError, match="finalize estimator") as exc,
    ):
        await budget_gate_prepare(_state([_result()]), deps)

    # Fail-closed diagnostics must not leak money figures.
    assert not any(character.isdigit() for character in str(exc.value))
    assert events == []


async def test_prepare_ignores_results_truncated_for_other_reasons(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = RecordingGateStore()
    deps = _deps(tmp_path, store=store, events=events, estimator=_constant_estimator("0.5"))

    with bind_budget(_ledger("0.1"), CostEstimator(None), _models()):
        result = await budget_gate_prepare(_state([_result(reason="timeout")]), deps)

    assert result["budget_gate"] is None
    assert result["finalize_estimate_usd"] == 0.5
    assert events == []
    assert store.record_calls == []


async def test_prepare_passes_through_when_remaining_equals_the_estimate(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = RecordingGateStore()
    deps = _deps(tmp_path, store=store, events=events, estimator=_constant_estimator("0.5"))

    with bind_budget(_ledger("0.5"), CostEstimator(None), _models()):
        result = await budget_gate_prepare(_state([]), deps)

    assert result["budget_gate"] is None
    assert result["budget_gate_seq"] is None
    assert result["finalize_estimate_usd"] == 0.5
    assert events == []
    assert store.record_calls == []


async def test_prepare_fails_closed_without_a_gate_store(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    deps = _deps(tmp_path, store=None, events=events, estimator=_constant_estimator("0.5"))

    with (
        bind_budget(_ledger("0.95"), CostEstimator(None), _models()),
        pytest.raises(RuntimeError, match="gate store"),
    ):
        await budget_gate_prepare(_state([_result()]), deps)

    assert events == []


async def test_prepare_propagates_estimator_errors(tmp_path) -> None:
    events: list[tuple[str, dict]] = []

    def failing_estimator(state):
        raise ValueError("missing price for the judge model")

    deps = _deps(tmp_path, store=RecordingGateStore(), events=events, estimator=failing_estimator)

    with (
        bind_budget(_ledger("0.95"), CostEstimator(None), _models()),
        pytest.raises(ValueError, match="missing price"),
    ):
        await budget_gate_prepare(_state([_result()]), deps)

    assert events == []


async def test_prepare_does_not_emit_when_record_reports_an_idempotent_duplicate(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = DuplicateGateStore()
    deps = _deps(tmp_path, store=store, events=events, estimator=_constant_estimator("0.5"))

    with bind_budget(_ledger("0.95"), CostEstimator(None), _models()):
        result = await budget_gate_prepare(_state([_result()]), deps)

    assert result["budget_gate_seq"] == 1
    assert result["budget_gate"] is not None
    assert events == []
    assert len(store.record_calls) == 1


async def test_prepare_reuses_a_pending_gate_row_by_existence_not_by_payload(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = RecordingGateStore()
    store.pending = SimpleNamespace(seq=4, payload_json={"remaining": 999.0})
    deps = _deps(tmp_path, store=store, events=events, estimator=_constant_estimator("0.5"))

    with bind_budget(_ledger("0.95"), CostEstimator(None), _models()):
        result = await budget_gate_prepare(_state([_result()]), deps)

    assert result["budget_gate_seq"] == 4
    assert result["budget_gate"] is not None
    assert events == []
    assert store.record_calls == []


def test_route_after_prepare_uses_the_gate_payload() -> None:
    gated = cast(EvalState, {"budget_gate": {"remaining": 0.1}})
    passed = cast(EvalState, {"budget_gate": None})

    assert _route_after_budget_gate_prepare(gated) == "budget_gate"
    assert _route_after_budget_gate_prepare(passed) == "merge"
