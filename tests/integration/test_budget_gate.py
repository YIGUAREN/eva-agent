"""Integration tests for the split budget gate: prepare records, pause resumes."""

import operator
from decimal import Decimal
from types import SimpleNamespace
from typing import Annotated, Any, TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

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
from eva_agent.graph.nodes.budget_gate import budget_gate_node, budget_gate_prepare
from eva_agent.graph.state import EvalState
from tests.support.runtime import dependencies

CONFIG: RunnableConfig = {"configurable": {"thread_id": "run-1"}}
BUDGET = Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5)


class GateState(TypedDict):
    run_id: str
    budget: Budget
    expert_results: Annotated[list[DimensionResult], operator.add]
    status: RunStatus
    finalize_estimate_usd: float
    finalize_overrun: bool
    budget_gate: dict | None
    budget_gate_seq: int | None
    rounds: int


class Task7GateStore:
    """Models ``BudgetDecisionRepository``: ``decide`` rewrites the action first."""

    def __init__(self) -> None:
        self.rows: dict[int, SimpleNamespace] = {}
        self.record_calls: list[tuple[int, str, dict[str, Any]]] = []

    async def next_seq(self, run_id: str) -> int:
        return max(self.rows, default=0) + 1

    async def record(self, run_id: str, seq: int, action: str, payload: dict) -> bool:
        self.record_calls.append((seq, action, payload))
        existing = self.rows.get(seq)
        if existing is not None:
            if existing.action == action and existing.payload_json == payload:
                return False
            raise RuntimeError("duplicate gate seq")
        self.rows[seq] = SimpleNamespace(seq=seq, action=action, payload_json=payload)
        return True

    async def latest_gate(self, run_id: str) -> SimpleNamespace | None:
        gates = [row for row in self.rows.values() if row.action == "gate"]
        return max(gates, key=lambda row: row.seq) if gates else None

    async def decide(self, run_id: str, seq: int, action: str, payload: dict) -> bool:
        row = self.rows.get(seq)
        if row is None or row.action != "gate":
            return False
        row.action = action
        row.payload_json = payload
        return True


class NoopLedgerStore:
    """A limited ledger requires a store; the gate tests do not persist."""

    async def load(self, run_id: str) -> None:
        return None

    async def save(self, run_id: str, usage: Any) -> None:
        return None


def _models() -> BudgetModels:
    model = ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")
    return BudgetModels(agent=model, judge=model, attack=model)


def _ledger(spent: str = "0", *, unlimited: bool = False) -> BudgetLedger:
    ledger = (
        BudgetLedger.unlimited(BUDGET)
        if unlimited
        else BudgetLedger(BUDGET, store=NoopLedgerStore())
    )
    if spent != "0":
        ledger._reserve_sync(Decimal(spent))
        ledger._settle_sync(Decimal(spent), Decimal(spent))
    return ledger


def _deps(
    tmp_path,
    *,
    events: list[tuple[str, dict]],
    store: Any,
    estimator: Any,
) -> GraphDependencies:
    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    base = dependencies(tmp_path, event_sink=sink)
    return GraphDependencies(
        planner=base.planner,
        tools=base.tools,
        catalog_ids=base.catalog_ids,
        data_dir=base.data_dir,
        max_input_bytes=base.max_input_bytes,
        max_dataset_rows=base.max_dataset_rows,
        default_sample_size=base.default_sample_size,
        event_sink=sink,
        finalize_estimator=estimator,
        gate_store=store,
    )


def _result(*, reason: str | None = "budget") -> DimensionResult:
    return DimensionResult(
        dimension=Dimension.QUALITY,
        status="partial",
        score=None,
        confidence=0.0,
        summary="预算触顶",
        usage=Usage(),
        truncated_reason=reason,
    )


def _initial(*, results: list[DimensionResult], rounds: int = 0) -> GateState:
    return {
        "run_id": "run-1",
        "budget": BUDGET,
        "expert_results": results,
        "status": RunStatus.RUNNING,
        "finalize_estimate_usd": 0.0,
        "finalize_overrun": False,
        "budget_gate": None,
        "budget_gate_seq": None,
        "rounds": rounds,
    }


def _route(state: GateState) -> str:
    return _route_after_budget_gate_prepare(cast(EvalState, state))


def _nodes(deps: GraphDependencies) -> tuple[Any, Any]:
    """Adapt the EvalState nodes to this test's smaller state schema."""

    async def prepare(state: GateState) -> dict:
        return await budget_gate_prepare(cast(EvalState, state), deps)

    async def pause(state: GateState) -> dict:
        return await budget_gate_node(cast(EvalState, state))

    return prepare, pause


def _seed(state: GateState) -> dict:
    return {"status": RunStatus.RUNNING}


def _merge(state: GateState) -> dict:
    return {"status": RunStatus.COMPLETED}


def _next_round(state: GateState) -> dict:
    return {"rounds": state["rounds"] + 1}


def _simple_graph(deps: GraphDependencies) -> Any:
    prepare, pause = _nodes(deps)
    graph = StateGraph(GateState)
    graph.add_node("seed", _seed)
    graph.add_node("budget_gate_prepare", prepare)
    graph.add_node("budget_gate", pause)
    graph.add_node("merge", _merge)
    graph.add_edge(START, "seed")
    graph.add_edge("seed", "budget_gate_prepare")
    graph.add_conditional_edges(
        "budget_gate_prepare",
        _route,
        {"budget_gate": "budget_gate", "merge": "merge"},
    )
    graph.add_edge("budget_gate", "merge")
    graph.add_edge("merge", END)
    return graph.compile(checkpointer=InMemorySaver())


def _looping_graph(deps: GraphDependencies) -> Any:
    prepare, pause = _nodes(deps)
    graph = StateGraph(GateState)
    graph.add_node("seed", _seed)
    graph.add_node("budget_gate_prepare", prepare)
    graph.add_node("budget_gate", pause)
    graph.add_node("round", _next_round)
    graph.add_node("merge", _merge)
    graph.add_edge(START, "seed")
    graph.add_edge("seed", "budget_gate_prepare")
    graph.add_conditional_edges(
        "budget_gate_prepare",
        _route,
        {"budget_gate": "budget_gate", "merge": "merge"},
    )
    graph.add_edge("budget_gate", "round")
    graph.add_conditional_edges(
        "round",
        lambda state: "budget_gate_prepare" if state["rounds"] < 2 else "merge",
        {"budget_gate_prepare": "budget_gate_prepare", "merge": "merge"},
    )
    graph.add_edge("merge", END)
    return graph.compile(checkpointer=InMemorySaver())


async def test_gate_interrupts_on_truncation_and_records_the_event(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = Task7GateStore()
    deps = _deps(tmp_path, events=events, store=store, estimator=lambda state: Decimal("0.5"))
    compiled = _simple_graph(deps)

    with bind_budget(_ledger("0.95"), CostEstimator(None), _models()):
        paused = await compiled.ainvoke(_initial(results=[_result()]), CONFIG)
        snapshot = await compiled.aget_state(CONFIG)

    assert snapshot.next == ("budget_gate",)
    assert [name for name, _ in events] == ["budget.gate"]
    assert events[0][1] == {
        "finalize_estimate": 0.5,
        "remaining": 0.05,
        "spent": 0.95,
        "truncated_dimensions": ["quality"],
    }
    assert store.rows[1].action == "gate"
    assert paused["budget_gate_seq"] == 1
    assert paused["finalize_overrun"] is False


async def test_action_rewrite_before_resume_keeps_one_row_and_one_event(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = Task7GateStore()
    deps = _deps(tmp_path, events=events, store=store, estimator=lambda state: Decimal("0.5"))
    compiled = _simple_graph(deps)

    with bind_budget(_ledger("0.95"), CostEstimator(None), _models()):
        await compiled.ainvoke(_initial(results=[_result()]), CONFIG)
        gate = await store.latest_gate("run-1")
        assert gate is not None
        assert gate.seq == 1
        # Task 7 rewrites the gate row before it resumes the graph.
        assert await store.decide(
            "run-1",
            1,
            "summarize",
            {**gate.payload_json, "decision": {"action": "summarize"}},
        )
        await compiled.ainvoke(
            Command(resume={"action": "summarize", "confirm_overrun": True}),
            CONFIG,
        )
        final = await compiled.aget_state(CONFIG)

    assert [name for name, _ in events] == ["budget.gate"]
    assert sorted(store.rows) == [1]
    assert store.rows[1].action == "summarize"
    assert final.values["budget_gate_seq"] == 1
    assert final.values["finalize_overrun"] is True
    assert final.values["status"] is RunStatus.COMPLETED
    # The decided row is no longer a pending gate.
    assert await store.latest_gate("run-1") is None


async def test_two_interrupts_get_distinct_seqs_and_one_event_each(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = Task7GateStore()
    deps = _deps(tmp_path, events=events, store=store, estimator=lambda state: Decimal("0.5"))
    compiled = _looping_graph(deps)
    ledger = _ledger("0.95")
    extended = Budget(max_cost_usd=2.0, max_wall_clock_s=60.0, max_target_requests=5)

    with bind_budget(ledger, CostEstimator(None), _models()):
        await compiled.ainvoke(_initial(results=[_result()]), CONFIG)
        first = await store.latest_gate("run-1")
        assert first is not None
        assert first.seq == 1
        assert await store.decide(
            "run-1",
            1,
            "extend",
            {**first.payload_json, "decision": {"action": "extend"}},
        )
        await compiled.ainvoke(
            Command(resume={"action": "extend", "budget": extended.model_dump(mode="json")}),
            CONFIG,
        )

        second = await store.latest_gate("run-1")
        assert second is not None
        assert second.seq == 2
        assert second.payload_json["remaining"] == 1.05
        assert await store.decide(
            "run-1",
            2,
            "summarize",
            {**second.payload_json, "decision": {"action": "summarize"}},
        )
        await compiled.ainvoke(
            Command(resume={"action": "summarize", "confirm_overrun": True}),
            CONFIG,
        )
        final = await compiled.aget_state(CONFIG)
        remaining = ledger.snapshot().remaining_usd

    assert [name for name, _ in events] == ["budget.gate", "budget.gate"]
    assert sorted(store.rows) == [1, 2]
    assert [store.rows[seq].action for seq in sorted(store.rows)] == ["extend", "summarize"]
    assert final.values["budget_gate_seq"] == 2
    assert final.values["status"] is RunStatus.COMPLETED
    assert remaining == Decimal("1.050000")
    assert await store.latest_gate("run-1") is None


async def test_gate_passes_without_interrupt_when_nothing_truncated_and_money_remains(
    tmp_path,
) -> None:
    events: list[tuple[str, dict]] = []
    store = Task7GateStore()
    deps = _deps(tmp_path, events=events, store=store, estimator=lambda state: Decimal("0.5"))
    compiled = _simple_graph(deps)

    with bind_budget(_ledger("0.1"), CostEstimator(None), _models()):
        paused = await compiled.ainvoke(_initial(results=[]), CONFIG)
        snapshot = await compiled.aget_state(CONFIG)

    assert snapshot.next == ()
    assert paused["status"] is RunStatus.COMPLETED
    assert paused["finalize_overrun"] is False
    assert events == []
    assert store.rows == {}


async def test_an_unlimited_ledger_bypasses_before_the_estimator(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = Task7GateStore()

    def failing_estimator(state):
        raise AssertionError("the estimator must not run when the ledger is unlimited")

    deps = _deps(tmp_path, events=events, store=store, estimator=failing_estimator)
    compiled = _simple_graph(deps)

    with bind_budget(_ledger(unlimited=True), CostEstimator(None), _models()):
        paused = await compiled.ainvoke(_initial(results=[_result()]), CONFIG)
        snapshot = await compiled.aget_state(CONFIG)

    assert snapshot.next == ()
    assert paused["status"] is RunStatus.COMPLETED
    assert paused["finalize_estimate_usd"] == 0.0
    assert paused["finalize_overrun"] is False
    assert events == []
    assert store.rows == {}


async def test_summarize_at_an_exact_estimate_is_not_an_overrun(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = Task7GateStore()
    deps = _deps(tmp_path, events=events, store=store, estimator=lambda state: Decimal("0.5"))
    compiled = _simple_graph(deps)

    with bind_budget(_ledger("0.5"), CostEstimator(None), _models()):
        await compiled.ainvoke(_initial(results=[_result()]), CONFIG)
        gate = await store.latest_gate("run-1")
        assert gate is not None
        assert gate.payload_json["remaining"] == 0.5
        assert await store.decide(
            "run-1",
            1,
            "summarize",
            {**gate.payload_json, "decision": {"action": "summarize"}},
        )
        await compiled.ainvoke(
            Command(resume={"action": "summarize", "confirm_overrun": False}),
            CONFIG,
        )
        final = await compiled.aget_state(CONFIG)

    assert final.values["finalize_overrun"] is False
    assert final.values["budget_gate_seq"] == 1
    assert [name for name, _ in events] == ["budget.gate"]


async def test_prepare_crash_replay_reuses_the_pending_seq_and_emits_once(tmp_path) -> None:
    events: list[tuple[str, dict]] = []
    store = Task7GateStore()
    deps = _deps(tmp_path, events=events, store=store, estimator=lambda state: Decimal("0.5"))
    state = cast(EvalState, _initial(results=[_result()]))

    with bind_budget(_ledger("0.95"), CostEstimator(None), _models()):
        first = await budget_gate_prepare(state, deps)
        # The gate row and its event were written before the crash, but the node
        # output was never checkpointed, so the graph replays the prepare node.
        assert await store.latest_gate("run-1") is not None
        second = await budget_gate_prepare(state, deps)

    assert first["budget_gate_seq"] == second["budget_gate_seq"] == 1
    assert [name for name, _ in events] == ["budget.gate"]
    assert sorted(store.rows) == [1]
    assert len(store.record_calls) == 1
