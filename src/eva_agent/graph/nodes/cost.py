"""Run the cost expert and record one dimension result."""

from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.experts.cost import CostExpert
from eva_agent.graph.state import ExpertState


async def cost_node(state: ExpertState, deps: GraphDependencies) -> dict:
    dimension_plan = state.get("dimension_plan")
    evidence = state.get("evidence")
    if dimension_plan is None or evidence is None:
        await deps.event_sink(
            state["run_id"],
            "cost.failed",
            {"error": "cost 专家缺少计划或证据"},
        )
        return {"status": RunStatus.FAILED, "errors": ["cost 专家缺少计划或证据"]}

    if dimension_plan.dimension is not Dimension.COST:
        await deps.event_sink(
            state["run_id"],
            "cost.failed",
            {"error": "cost 专家收到错误的维度"},
        )
        return {"status": RunStatus.FAILED, "errors": ["cost 专家收到错误的维度"]}

    if state.get("probe") is None:
        await deps.event_sink(
            state["run_id"],
            "cost.failed",
            {"error": "cost 专家缺少 probe 摘要"},
        )
        return {"status": RunStatus.FAILED, "errors": ["cost 专家缺少 probe 摘要"]}

    expert = CostExpert(deps.tools)
    try:
        result = await expert.run(
            run_id=state["run_id"],
            plan=dimension_plan,
            evidence=evidence,
        )
    except Exception as exc:
        result = DimensionResult(
            dimension=Dimension.COST,
            status="failed",
            score=None,
            confidence=0.0,
            summary="cost 专家执行失败",
            usage=Usage(),
            error=str(exc),
        )

    if result.status == "failed":
        await deps.event_sink(
            state["run_id"],
            "cost.failed",
            {"error": result.error, "summary": result.summary},
        )
    else:
        await deps.event_sink(
            state["run_id"],
            "cost.completed",
            {"status": result.status, "score": result.score},
        )

    return {"expert_results": [result]}
