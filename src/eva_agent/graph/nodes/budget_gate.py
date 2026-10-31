"""Pause before merge when the budget cannot cover the finalize tail."""

from __future__ import annotations

from decimal import Decimal

from langgraph.types import interrupt

from eva_agent.budget.context import budget_context
from eva_agent.budget.types import quantize_money
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.state import EvalState


def _pass_through(estimate: Decimal) -> dict:
    return {
        "status": RunStatus.RUNNING,
        "finalize_estimate_usd": float(estimate),
        "finalize_overrun": False,
        "budget_gate": None,
        "budget_gate_seq": None,
    }


async def budget_gate_node(state: EvalState, deps: GraphDependencies) -> dict:
    run_id = state["run_id"]
    truncated = sorted(
        {
            result.dimension.value
            for result in state.get("expert_results") or []
            if result.truncated_reason == "budget"
        }
    )
    context = budget_context()
    snapshot = context.ledger.snapshot() if context is not None else None
    remaining = snapshot.remaining_usd if snapshot is not None else Decimal("0")
    spent = snapshot.spent_usd if snapshot is not None else Decimal("0")
    estimate = (
        deps.finalize_estimator(state) if deps.finalize_estimator is not None else Decimal("0")
    )
    estimate = quantize_money(estimate)

    # An unbound or unlimited ledger has no limits to pause on, so the gate must
    # never interrupt a run whose budget is not enforced.
    if context is None or not context.ledger.enforced:
        return _pass_through(estimate)

    if not truncated and remaining >= estimate:
        return _pass_through(estimate)

    store = deps.gate_store
    if store is None:
        raise RuntimeError("budget gate requires a gate store")
    payload = {
        "finalize_estimate": float(estimate),
        "remaining": float(remaining),
        "spent": float(spent),
        "truncated_dimensions": truncated,
    }
    # Re-running the node after an interrupt must reuse the same seq, otherwise
    # a crash between the interrupt and the decision would create two gates.
    latest = await store.latest_gate(run_id)
    if latest is not None and latest.payload_json == payload:
        seq = int(latest.seq)
    else:
        seq = await store.next_seq(run_id)
        await store.record(run_id, seq, "gate", payload)
        await deps.event_sink(run_id, "budget.gate", payload)

    decision = interrupt(payload)
    if not isinstance(decision, dict):
        raise ValueError("budget decision must be an object")
    action = decision.get("action")
    if action == "extend":
        budget = Budget.model_validate(decision.get("budget"))
        if context is not None:
            context.ledger.update_budget(budget)
        return {
            "budget": budget,
            "status": RunStatus.RUNNING,
            "finalize_estimate_usd": float(estimate),
            "finalize_overrun": False,
            "budget_gate": payload,
            "budget_gate_seq": seq,
        }
    if action == "summarize":
        return {
            "status": RunStatus.RUNNING,
            "finalize_estimate_usd": float(estimate),
            "finalize_overrun": remaining < estimate,
            "budget_gate": payload,
            "budget_gate_seq": seq,
        }
    raise ValueError(f"unknown budget decision: {action!r}")
