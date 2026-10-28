"""Run the quality expert and record one dimension result."""

from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.experts.quality import QualityExpert
from eva_agent.graph.state import EvalState


async def quality_node(state: EvalState, deps: GraphDependencies) -> dict:
    plan = state["plan"]
    evidence = state["evidence"]
    if plan is None or evidence is None:
        return {"status": RunStatus.FAILED, "errors": ["quality 专家缺少计划或证据"]}

    dimension_plan = next(
        (item for item in plan.dimensions if item.enabled and item.dimension is Dimension.QUALITY),
        None,
    )
    if dimension_plan is None:
        return {"status": RunStatus.FAILED, "errors": ["计划中没有启用的 quality 维度"]}

    expert = QualityExpert(deps.tools, default_sample_size=deps.default_sample_size)
    try:
        result = await expert.run(
            run_id=state["run_id"],
            plan=dimension_plan,
            evidence=evidence,
        )
    except Exception as exc:
        result = DimensionResult(
            dimension=Dimension.QUALITY,
            status="failed",
            score=None,
            confidence=0.0,
            summary="quality 专家执行失败",
            usage=Usage(),
            error=str(exc),
        )
        await deps.event_sink(
            state["run_id"],
            "quality.failed",
            {"error": str(exc)},
        )
    else:
        await deps.event_sink(
            state["run_id"],
            "quality.completed",
            {"status": result.status, "score": result.score},
        )

    return {"expert_results": [result]}
