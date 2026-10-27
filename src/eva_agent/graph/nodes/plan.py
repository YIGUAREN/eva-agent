"""Ask the planner for an evaluation plan."""

from eva_agent.domain.run import RunStatus
from eva_agent.graph.planner import Planner
from eva_agent.graph.state import EvalState


async def plan_node(state: EvalState, planner: Planner) -> dict:
    evidence = state["evidence"]
    if evidence is None:
        return {"status": RunStatus.FAILED, "errors": ["缺少证据，无法制定计划"]}

    try:
        plan = await planner.plan(evidence, state["budget"])
    except Exception as exc:
        return {"status": RunStatus.FAILED, "errors": [f"制定评估计划失败: {exc}"]}
    return {"plan": plan, "status": RunStatus.COMPLETED}
