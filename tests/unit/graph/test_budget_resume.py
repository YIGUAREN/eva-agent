from typing import Literal, cast

import pytest
from langgraph.graph import END

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.graph.build import route_after_budget_gate
from eva_agent.graph.nodes import budget_gate as budget_gate_module
from eva_agent.graph.nodes.budget_gate import budget_gate_node
from eva_agent.graph.state import EvalState, merge_expert_results

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
