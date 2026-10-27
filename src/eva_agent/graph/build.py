"""Compile the phase-0 graph."""

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from eva_agent.graph.nodes.intake import intake_node
from eva_agent.graph.nodes.plan import plan_node
from eva_agent.graph.planner import Planner
from eva_agent.graph.state import EvalState


def build_graph(planner: Planner, checkpointer: Any | None = None) -> Any:
    graph = StateGraph(EvalState)
    graph.add_node("intake", intake_node)
    graph.add_node("plan", partial(plan_node, planner=planner))
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "plan")
    graph.add_edge("plan", END)
    return graph.compile(checkpointer=checkpointer)
