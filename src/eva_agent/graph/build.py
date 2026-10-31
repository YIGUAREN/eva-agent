"""Compile the evaluation graph (phase 1 quality, phase 2 target experts)."""

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from eva_agent.domain.plan import Dimension
from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.domain.target import TargetKind
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.nodes.analyze import analyze_node
from eva_agent.graph.nodes.approve import approve_node
from eva_agent.graph.nodes.budget_gate import budget_gate_node, budget_gate_prepare
from eva_agent.graph.nodes.cost import cost_node
from eva_agent.graph.nodes.dispatch import dispatch_node
from eva_agent.graph.nodes.intake import intake_node
from eva_agent.graph.nodes.latency import latency_node
from eva_agent.graph.nodes.merge import merge_node
from eva_agent.graph.nodes.plan import plan_node
from eva_agent.graph.nodes.probe import probe_node
from eva_agent.graph.nodes.quality import quality_node
from eva_agent.graph.nodes.report import report_node
from eva_agent.graph.nodes.security import security_node
from eva_agent.graph.state import EvalState

SUPPORTED_DIMENSIONS = frozenset({Dimension.QUALITY, Dimension.LATENCY, Dimension.COST, Dimension.SECURITY})


def _route_after_intake(state: EvalState) -> str:
    return END if state["status"] is RunStatus.FAILED else "plan"


def _route_after_plan(state: EvalState) -> str:
    return "approve" if state["status"] is RunStatus.WAITING_APPROVAL else END


def _route_after_approve(state: EvalState) -> str:
    if state["status"] is not RunStatus.RUNNING:
        return END
    plan = state["plan"]
    # An all-skipped plan has nothing to probe; probe_node would reject it.
    if plan is None or not any(item.enabled for item in plan.dimensions):
        return "dispatch"
    if state["input"].kind is not InputKind.TARGET:
        return "dispatch"
    evidence = state.get("evidence")
    target = evidence.target if evidence is not None else None
    # agent_scan runs entirely inside AIG; a local probe would bill the target
    # twice and would produce rows no expert consumes.
    if target is not None and target.kind is TargetKind.AGENT_HTTP:
        return "dispatch"
    return "probe"


def _route_after_budget_gate_prepare(state: EvalState) -> str:
    # The prepare node clears the payload on a pass-through, so its presence is
    # the single signal that a pause is required.
    return "budget_gate" if state.get("budget_gate") is not None else "merge"


def route_experts(state: EvalState) -> list[Send] | str:
    if state["status"] is not RunStatus.RUNNING or state["plan"] is None:
        return END
    sends = [
        Send(
            item.dimension.value,
            {
                "run_id": state["run_id"],
                "dimension_plan": item,
                "evidence": state["evidence"],
                "probe": state.get("probe"),
                "budget": state["budget"],
                "authorization_ref": state.get("authorization_ref"),
                "requester_is_admin": state.get("requester_is_admin"),
            },
        )
        for item in state["plan"].dimensions
        if item.enabled and item.dimension in SUPPORTED_DIMENSIONS
    ]
    return sends or "merge"


def build_graph(dependencies: GraphDependencies, checkpointer: Any | None = None) -> Any:
    graph = StateGraph(EvalState)
    graph.add_node("intake", partial(intake_node, deps=dependencies))
    graph.add_node("plan", partial(plan_node, deps=dependencies))
    graph.add_node("approve", partial(approve_node, deps=dependencies))
    graph.add_node("probe", partial(probe_node, deps=dependencies))
    graph.add_node("dispatch", partial(dispatch_node, deps=dependencies))
    graph.add_node("quality", partial(quality_node, deps=dependencies))
    graph.add_node("latency", partial(latency_node, deps=dependencies))
    graph.add_node("cost", partial(cost_node, deps=dependencies))
    graph.add_node("security", partial(security_node, deps=dependencies))
    graph.add_node("budget_gate_prepare", partial(budget_gate_prepare, deps=dependencies))
    graph.add_node("budget_gate", budget_gate_node)
    graph.add_node("merge", partial(merge_node, deps=dependencies))
    graph.add_node("analyze", partial(analyze_node, deps=dependencies))
    graph.add_node("report", partial(report_node, deps=dependencies))

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", _route_after_intake, {"plan": "plan", END: END})
    graph.add_conditional_edges("plan", _route_after_plan, {"approve": "approve", END: END})
    graph.add_conditional_edges(
        "approve",
        _route_after_approve,
        {"probe": "probe", "dispatch": "dispatch", END: END},
    )
    graph.add_edge("probe", "dispatch")
    graph.add_conditional_edges(
        "dispatch",
        route_experts,
        {
            "quality": "quality",
            "latency": "latency",
            "cost": "cost",
            "security": "security",
            "merge": "merge",
            END: END,
        },
    )
    graph.add_edge("quality", "budget_gate_prepare")
    graph.add_edge("latency", "budget_gate_prepare")
    graph.add_edge("cost", "budget_gate_prepare")
    graph.add_edge("security", "budget_gate_prepare")
    graph.add_conditional_edges(
        "budget_gate_prepare",
        _route_after_budget_gate_prepare,
        {"budget_gate": "budget_gate", "merge": "merge"},
    )
    graph.add_edge("budget_gate", "merge")
    graph.add_edge("merge", "analyze")
    graph.add_edge("analyze", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)
