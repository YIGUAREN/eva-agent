"""Ask the planner for an evaluation plan."""

from eva_agent.domain.run import RunStatus
from eva_agent.graph.planner import Planner
from eva_agent.graph.state import EvalState


async def plan_node(state: EvalState, planner: Planner) -> dict:
    evidence = state["evidence"]
    if evidence is None:
        return {"status": RunStatus.FAILED, "errors": ["缺少证据，无法制定计划"]}

    plan = await planner.plan(evidence, state["budget"])
    return {"plan": plan, "status": RunStatus.COMPLETED}
