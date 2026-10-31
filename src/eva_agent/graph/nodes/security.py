"""Run the security expert and record one dimension result."""

from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.experts.security import SecurityExpert
from eva_agent.graph.state import ExpertState


async def security_node(state: ExpertState, deps: GraphDependencies) -> dict:
    dimension_plan = state.get("dimension_plan")
    evidence = state.get("evidence")
    if dimension_plan is None or evidence is None:
        await deps.event_sink(state["run_id"], "security.failed", {"error": "security 专家缺少计划或证据"})
        return {"status": RunStatus.FAILED, "errors": ["security 专家缺少计划或证据"]}

    if dimension_plan.dimension is not Dimension.SECURITY:
        await deps.event_sink(state["run_id"], "security.failed", {"error": "security 专家收到错误的维度"})
        return {"status": RunStatus.FAILED, "errors": ["security 专家收到错误的维度"]}

    try:
        result = await SecurityExpert(deps).run(
            run_id=state["run_id"],
            plan=dimension_plan,
            evidence=evidence,
            probe=state.get("probe"),
            budget=state["budget"],
            authorization_ref=state.get("authorization_ref"),
            requester_is_admin=state.get("requester_is_admin"),
            selected=state.get("graders"),
        )
    except Exception as exc:
        result = DimensionResult(
            dimension=Dimension.SECURITY,
            status="failed",
            score=None,
            confidence=0.0,
            summary="security 专家执行失败",
            usage=Usage(),
            error=type(exc).__name__,
        )

    if result.status == "failed":
        await deps.event_sink(
            state["run_id"],
            "security.failed",
            {"error": result.error or "security 专家执行失败"},
        )
    else:
        await deps.event_sink(
            state["run_id"],
            "security.completed",
            {"status": result.status, "score": result.score},
        )
    return {"expert_results": [result]}
