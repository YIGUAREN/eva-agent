from dataclasses import replace
from decimal import Decimal
from typing import Literal, cast

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END
from langgraph.types import Command

from eva_agent.adapters.target import tokens as tokens_module
from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.graph.build import build_graph, route_after_budget_gate
from eva_agent.graph.nodes import budget_gate as budget_gate_module
from eva_agent.graph.nodes.budget_gate import budget_gate_node
from eva_agent.graph.state import EvalState, merge_expert_results
from eva_agent.pricing import ModelPrice, PricingTable
from tests.support.runtime import (
    FakePlanner,
    FakeRunner,
    dependencies,
    fake_config,
    initial_state,
    paste_input,
    sample_plan,
)

CONFIG = {"configurable": {"thread_id": "run-1"}}

ResultStatus = Literal["ok", "partial", "skipped", "failed"]


def _result(dimension: Dimension, status: ResultStatus, *, truncated: str | None = None) -> DimensionResult:
    return DimensionResult(
        dimension=dimension,
        status=status,
        score=0.5,
        confidence=0.5,
        summary="s",
        usage=Usage(),
        truncated_reason=truncated,
    )


def test_a_new_result_replaces_the_same_dimension() -> None:
    current = [_result(Dimension.QUALITY, "partial", truncated="budget")]
    update = [_result(Dimension.QUALITY, "ok")]
    merged = merge_expert_results(current, update)
    assert len(merged) == 1
    assert merged[0].status == "ok"
    assert merged[0].truncated_reason is None


def _plan() -> EvalPlan:
    return EvalPlan(
        objective="o",
        evidence_summary="e",
        dimensions=[
            DimensionPlan(dimension=Dimension.QUALITY, enabled=True, sample_size=1, graders=[]),
            DimensionPlan(dimension=Dimension.LATENCY, enabled=True, sample_size=1, graders=[]),
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.COST, reason="phase 1"),
            SkippedDimension(dimension=Dimension.SECURITY, reason="phase 1"),
        ],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1),
    )


def _state(retry: list[str]) -> EvalState:
    plan = _plan()
    return cast(
        EvalState,
        {
            "run_id": "run-1",
            "status": RunStatus.RUNNING,
            "plan": plan,
            "evidence": EvidenceBundle(
                kind=InputKind.PASTE,
                pasted={"query": "q", "response": "a"},
                manifest_sha256="0" * 64,
            ),
            "probe": None,
            "budget": plan.budget,
            "retry_dimensions": retry,
        },
    )


def test_extend_retries_only_the_truncated_dimensions() -> None:
    sends = route_after_budget_gate(_state(["quality"]))
    assert not isinstance(sends, str)
    assert [send.node for send in sends] == ["quality"]
    assert route_after_budget_gate(_state([])) == "merge"


def test_a_finished_run_routes_to_end() -> None:
    state = cast(EvalState, {**_state(["quality"]), "status": RunStatus.COMPLETED})
    assert route_after_budget_gate(state) == END


async def test_extend_returns_the_truncated_dimensions_from_the_gate_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extended = Budget(max_cost_usd=2.0, max_wall_clock_s=60.0, max_target_requests=5)
    monkeypatch.setattr(
        budget_gate_module,
        "interrupt",
        lambda payload: {"action": "extend", "budget": extended.model_dump(mode="json")},
    )
    model = ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")
    state = cast(
        EvalState,
        {
            "run_id": "run-1",
            "budget": extended,
            "status": RunStatus.RUNNING,
            "budget_gate": {
                "finalize_estimate": 0.5,
                "remaining": 0.1,
                "spent": 0.9,
                "truncated_dimensions": ["quality"],
            },
            "budget_gate_seq": 1,
        },
    )

    with bind_budget(
        BudgetLedger.unlimited(extended),
        CostEstimator(None),
        BudgetModels(agent=model, judge=model, attack=model),
    ):
        result = await budget_gate_node(state)

    assert result["retry_dimensions"] == ["quality"]


async def test_summarize_clears_the_retry_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        budget_gate_module,
        "interrupt",
        lambda payload: {"action": "summarize", "confirm_overrun": True},
    )
    state = cast(
        EvalState,
        {
            "run_id": "run-1",
            "budget": _plan().budget,
            "status": RunStatus.RUNNING,
            "budget_gate": {
                "finalize_estimate": 0.5,
                "remaining": 0.1,
                "spent": 0.9,
                "truncated_dimensions": ["quality"],
            },
            "budget_gate_seq": 1,
        },
    )

    result = await budget_gate_node(state)

    assert result["retry_dimensions"] == []


def test_a_retry_with_no_enabled_dimension_routes_to_merge() -> None:
    # COST is skipped, so no enabled plan dimension matches the retry list.
    assert route_after_budget_gate(_state(["cost"])) == "merge"


def test_a_retry_outside_supported_dimensions_routes_to_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eva_agent.graph.build.SUPPORTED_DIMENSIONS",
        frozenset({Dimension.LATENCY}),
    )

    assert route_after_budget_gate(_state(["quality"])) == "merge"


class _FakeEncoding:
    """Offline stand-in for ``tiktoken.Encoding``: one token per character."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


@pytest.fixture
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch):
    tokens_module._encoding.cache_clear()
    monkeypatch.setattr(tokens_module.tiktoken, "get_encoding", lambda name: _FakeEncoding())
    yield
    tokens_module._encoding.cache_clear()


class _NoopLedgerStore:
    async def load(self, run_id: str) -> None:
        return None

    async def save(self, run_id: str, usage: object) -> None:
        return None


class _RecordingGateStore:
    def __init__(self) -> None:
        self.rows: list[tuple[int, dict]] = []

    async def next_seq(self, run_id: str) -> int:
        return len(self.rows) + 1

    async def record(self, run_id: str, seq: int, action: str, payload: dict) -> bool:
        self.rows.append((seq, payload))
        return True

    async def latest_gate(self, run_id: str):
        return None


def _resume_plan() -> EvalPlan:
    return sample_plan(sample_size=1).model_copy(
        update={
            "dimensions": [
                DimensionPlan(
                    dimension=Dimension.QUALITY,
                    enabled=True,
                    sample_size=1,
                    graders=[GraderSpec(level="builtin", name="relevance", builtin_id="relevance")],
                ),
                DimensionPlan(dimension=Dimension.SECURITY, enabled=True, sample_size=1, graders=[]),
            ],
            "skipped": [
                SkippedDimension(dimension=Dimension.LATENCY, reason="not applicable to paste input"),
                SkippedDimension(dimension=Dimension.COST, reason="not applicable to paste input"),
            ],
        }
    )


def _priced_estimator() -> CostEstimator:
    return CostEstimator(
        PricingTable(
            [
                ModelPrice(
                    provider="openai-compatible",
                    model="fake",
                    input_per_million=Decimal("1000"),
                    output_per_million=Decimal("1000"),
                )
            ]
        )
    )


async def test_the_compiled_graph_retries_the_truncated_experts_after_extend(
    tmp_path,
    fake_tokenizer,
) -> None:
    runner = FakeRunner()
    deps = replace(
        dependencies(tmp_path, planner=FakePlanner(plan=_resume_plan()), runner=runner),
        finalize_estimator=lambda state: Decimal("0.5"),
        gate_store=_RecordingGateStore(),
    )
    app = build_graph(deps, MemorySaver())
    model = fake_config(tmp_path).models.judge
    ledger = BudgetLedger(
        Budget(max_cost_usd=0.000001, max_wall_clock_s=60.0, max_target_requests=90),
        store=_NoopLedgerStore(),
    )
    extended = Budget(max_cost_usd=100.0, max_wall_clock_s=60.0, max_target_requests=90)

    with bind_budget(ledger, _priced_estimator(), BudgetModels(agent=model, judge=model, attack=model)):
        await app.ainvoke(initial_state(paste_input()), CONFIG)
        await app.ainvoke(Command(resume={"approved": True}), CONFIG)
        paused = await app.aget_state(CONFIG)

        assert paused.next == ("budget_gate",)
        assert runner.calls == 0
        before = paused.values["expert_results"]
        assert {(result.dimension, result.status, result.truncated_reason) for result in before} == {
            (Dimension.QUALITY, "partial", "budget"),
            (Dimension.SECURITY, "partial", "budget"),
        }

        await app.ainvoke(
            Command(resume={"action": "extend", "budget": extended.model_dump(mode="json")}),
            CONFIG,
        )
        final = await app.aget_state(CONFIG)

    assert final.values["status"] is RunStatus.COMPLETED
    results = final.values["expert_results"]
    assert len(results) == 2
    assert {result.dimension for result in results} == {Dimension.QUALITY, Dimension.SECURITY}
    assert all(result.status == "ok" and result.truncated_reason is None for result in results)
    # The quality relevance spec and the security harmfulness spec ran on the retry.
    assert runner.calls == 2
