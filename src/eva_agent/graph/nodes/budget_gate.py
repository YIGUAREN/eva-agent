"""Prepare a budget gate before merge, then pause for the operator's decision.

The node is split so that LangGraph's resume replay never re-records the gate:
``budget_gate_prepare`` runs once on the expert fan-in (scan, estimate, seq,
record, event), and ``budget_gate`` only wraps ``interrupt`` and applies the
decision.  Replaying the pause node therefore makes no store calls and emits no
event, even when the repository has rewritten the gate row's action.
"""

from __future__ import annotations

from decimal import Decimal

from langgraph.types import interrupt

from eva_agent.budget.context import BudgetContext, budget_context
from eva_agent.budget.types import quantize_money
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.state import EvalState

ZERO = Decimal("0")


def _pass_through(estimate: Decimal) -> dict:
    return {
        "status": RunStatus.RUNNING,
        "finalize_estimate_usd": float(estimate),
        "finalize_overrun": False,
        "budget_gate": None,
        "budget_gate_seq": None,
        "retry_dimensions": [],
    }


def _require_context() -> BudgetContext:
    context = budget_context()
    if context is None:
        raise RuntimeError("budget gate resume requires a bound budget context")
    return context


async def budget_gate_prepare(state: EvalState, deps: GraphDependencies) -> dict:
    run_id = state["run_id"]
    truncated = sorted(
        {
            result.dimension.value
            for result in state.get("expert_results") or []
            if result.truncated_reason == "budget"
        }
    )
    context = budget_context()
    # An unbound or unlimited ledger has no limits to pause on.  Bypass before
    # the estimator so a run without pricing never fails on a missing price.
    if context is None or not context.ledger.enforced:
        return _pass_through(ZERO)

    snapshot = context.ledger.snapshot()
    remaining = snapshot.remaining_usd
    spent = snapshot.spent_usd
    estimator = deps.finalize_estimator
    if estimator is None:
        raise RuntimeError("budget gate requires a finalize estimator")
    estimate = quantize_money(estimator(state))

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
    # A replay may find the row written by an attempt that crashed before its
    # output was checkpointed.  Reuse it by existence: the payload can differ
    # (an extension changes ``remaining``) and Task 7 rewrites ``action``.
    latest = await store.latest_gate(run_id)
    if latest is not None:
        seq = int(latest.seq)
    else:
        seq = await store.next_seq(run_id)
        created = await store.record(run_id, seq, "gate", payload)
        if created:
            await deps.event_sink(run_id, "budget.gate", payload)
    return {
        "status": RunStatus.RUNNING,
        "finalize_estimate_usd": float(estimate),
        "finalize_overrun": False,
        "budget_gate": payload,
        "budget_gate_seq": seq,
        "retry_dimensions": [],
    }


async def budget_gate_node(state: EvalState) -> dict:
    seq = state.get("budget_gate_seq")
    payload = state.get("budget_gate")
    if seq is None or payload is None:
        raise RuntimeError("budget gate pause requires a prepared gate")

    decision = interrupt(payload)
    if not isinstance(decision, dict):
        raise ValueError("budget decision must be an object")
    action = decision.get("action")
    estimate = Decimal(str(payload["finalize_estimate"]))
    if action == "extend":
        budget = Budget.model_validate(decision.get("budget"))
        _require_context().ledger.update_budget(budget)
        truncated = payload.get("truncated_dimensions")
        return {
            "budget": budget,
            "status": RunStatus.RUNNING,
            "finalize_estimate_usd": float(estimate),
            "finalize_overrun": False,
            "budget_gate": payload,
            "budget_gate_seq": seq,
            "retry_dimensions": truncated if isinstance(truncated, list) else [],
        }
    if action == "summarize":
        remaining = Decimal(str(payload["remaining"]))
        return {
            "status": RunStatus.RUNNING,
            "finalize_estimate_usd": float(estimate),
            "finalize_overrun": remaining < estimate,
            "budget_gate": payload,
            "budget_gate_seq": seq,
            "retry_dimensions": [],
        }
    raise ValueError(f"unknown budget decision: {action!r}")
