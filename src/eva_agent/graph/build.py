"""Compile the phase-1 evaluation graph."""

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.nodes.analyze import analyze_node
from eva_agent.graph.nodes.approve import approve_node
from eva_agent.graph.nodes.dispatch import dispatch_node
from eva_agent.graph.nodes.intake import intake_node
from eva_agent.graph.nodes.merge import merge_node
from eva_agent.graph.nodes.plan import plan_node
from eva_agent.graph.nodes.quality import quality_node
from eva_agent.graph.nodes.report import report_node
from eva_agent.graph.state import EvalState


def _route_after_intake(state: EvalState) -> str:
    return END if state["status"] is RunStatus.FAILED else "plan"


def _route_after_plan(state: EvalState) -> str:
    return "approve" if state["status"] is RunStatus.WAITING_APPROVAL else END


def _route_after_approve(state: EvalState) -> str:
    return "dispatch" if state["status"] is RunStatus.RUNNING else END


def _route_after_dispatch(state: EvalState) -> str:
    return "quality" if state["status"] is RunStatus.RUNNING else END


def build_graph(dependencies: GraphDependencies, checkpointer: Any | None = None) -> Any:
    graph = StateGraph(EvalState)
    graph.add_node("intake", partial(intake_node, deps=dependencies))
    graph.add_node("plan", partial(plan_node, deps=dependencies))
    graph.add_node("approve", partial(approve_node, deps=dependencies))
    graph.add_node("dispatch", partial(dispatch_node, deps=dependencies))
    graph.add_node("quality", partial(quality_node, deps=dependencies))
    graph.add_node("merge", partial(merge_node, deps=dependencies))
    graph.add_node("analyze", partial(analyze_node, deps=dependencies))
    graph.add_node("report", partial(report_node, deps=dependencies))

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", _route_after_intake, {"plan": "plan", END: END})
    graph.add_conditional_edges("plan", _route_after_plan, {"approve": "approve", END: END})
    graph.add_conditional_edges("approve", _route_after_approve, {"dispatch": "dispatch", END: END})
    graph.add_conditional_edges("dispatch", _route_after_dispatch, {"quality": "quality", END: END})
    graph.add_edge("quality", "merge")
    graph.add_edge("merge", "analyze")
    graph.add_edge("analyze", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)
