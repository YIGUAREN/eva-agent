import operator
from decimal import Decimal
from functools import partial
from types import SimpleNamespace
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.budget.types import RunUsage
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.nodes.budget_gate import budget_gate_node
from tests.support.runtime import dependencies

CONFIG: RunnableConfig = {"configurable": {"thread_id": "run-1"}}


class GateState(TypedDict):
    run_id: str
    budget: Budget
    expert_results: Annotated[list[DimensionResult], operator.add]
    status: RunStatus
    finalize_estimate_usd: float
    finalize_overrun: bool
    budget_gate: dict | None
    budget_gate_seq: int | None


class FakeGateStore:
    def __init__(self) -> None:
        self.rows: dict[int, SimpleNamespace] = {}

    async def next_seq(self, run_id: str) -> int:
        return max(self.rows, default=0) + 1

    async def record(self, run_id: str, seq: int, action: str, payload: dict) -> bool:
        self.rows[seq] = SimpleNamespace(seq=seq, action=action, payload_json=payload)
        return True

    async def latest_gate(self, run_id: str):
        gates = [row for row in self.rows.values() if row.action == "gate"]
        return max(gates, key=lambda row: row.seq) if gates else None


class NoopLedgerStore:
    """A limited ledger requires a store; the gate tests do not persist."""

    async def load(self, run_id: str) -> RunUsage | None:
        return None

    async def save(self, run_id: str, usage: RunUsage) -> None:
        return None


def truncated_quality() -> DimensionResult:
    return DimensionResult(
        dimension=Dimension.QUALITY,
        status="partial",
        score=None,
        confidence=0.0,
        summary="预算触顶",
        usage=Usage(),
        truncated_reason="budget",
    )


async def _run_gate(
    tmp_path,
    *,
    truncated: bool,
    spent: str = "0.95",
    unlimited: bool = False,
    resume: bool = False,
):
    events: list[tuple[str, dict]] = []

    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    base = dependencies(tmp_path, event_sink=sink)
    store = FakeGateStore()
    deps = GraphDependencies(
        planner=base.planner,
        tools=base.tools,
        catalog_ids=base.catalog_ids,
        data_dir=base.data_dir,
        max_input_bytes=base.max_input_bytes,
        max_dataset_rows=base.max_dataset_rows,
        default_sample_size=base.default_sample_size,
        event_sink=sink,
        finalize_estimator=lambda state: Decimal("0.5"),
        gate_store=store,
    )
    graph = StateGraph(GateState)
    graph.add_node("seed", lambda state: {"status": RunStatus.RUNNING})
    graph.add_node("budget_gate", partial(budget_gate_node, deps=deps))
    graph.add_node("merge", lambda state: {"status": RunStatus.COMPLETED})
    graph.add_edge(START, "seed")
    graph.add_edge("seed", "budget_gate")
    graph.add_edge("budget_gate", "merge")
    graph.add_edge("merge", END)
    compiled = graph.compile(checkpointer=InMemorySaver())

    model = ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")
    budget = Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=5)
    ledger = BudgetLedger.unlimited(budget) if unlimited else BudgetLedger(budget, store=NoopLedgerStore())
    ledger._reserve_sync(Decimal(spent))
    ledger._settle_sync(Decimal(spent), Decimal(spent))
    initial: GateState = {
        "run_id": "run-1",
        "budget": budget,
        "expert_results": [truncated_quality()] if truncated else [],
        "status": RunStatus.RUNNING,
        "finalize_estimate_usd": 0.0,
        "finalize_overrun": False,
        "budget_gate": None,
        "budget_gate_seq": None,
    }
    with bind_budget(ledger, CostEstimator(None), BudgetModels(agent=model, judge=model, attack=model)):
        paused = await compiled.ainvoke(initial, CONFIG)
        snapshot = await compiled.aget_state(CONFIG)
        if not resume or not snapshot.next:
            return paused, snapshot, events, store, CONFIG, compiled
        await compiled.ainvoke(
            Command(resume={"action": "summarize", "confirm_overrun": True}),
            CONFIG,
        )
    final = await compiled.aget_state(CONFIG)
    return paused, final, events, store, CONFIG, compiled


async def test_gate_interrupts_on_truncation_and_records_the_event(tmp_path) -> None:
    paused, snapshot, events, store, config, compiled = await _run_gate(tmp_path, truncated=True)
    assert snapshot.next == ("budget_gate",)
    assert [name for name, _ in events] == ["budget.gate"]
    assert events[0][1] == {
        "finalize_estimate": 0.5,
        "remaining": 0.05,
        "spent": 0.95,
        "truncated_dimensions": ["quality"],
    }
    assert store.rows[1].action == "gate"
    assert paused["finalize_overrun"] is False


async def test_resume_summarize_marks_the_overrun(tmp_path) -> None:
    _, final, _, _, _, _ = await _run_gate(tmp_path, truncated=True, resume=True)
    assert final.values["status"] is RunStatus.COMPLETED
    assert final.values["finalize_overrun"] is True
    assert final.values["budget_gate_seq"] == 1


async def test_gate_passes_without_interrupt_when_nothing_truncated_and_money_remains(tmp_path) -> None:
    paused, snapshot, events, _, _, _ = await _run_gate(tmp_path, truncated=False, spent="0.1")
    assert snapshot.next == ()
    assert paused["status"] is RunStatus.COMPLETED
    assert paused["finalize_overrun"] is False
    assert events == []


async def test_an_unlimited_ledger_bypasses_the_gate(tmp_path) -> None:
    paused, snapshot, events, store, _, _ = await _run_gate(tmp_path, truncated=True, unlimited=True)
    assert snapshot.next == ()
    assert paused["status"] is RunStatus.COMPLETED
    assert paused["finalize_overrun"] is False
    assert events == []
    assert store.rows == {}
