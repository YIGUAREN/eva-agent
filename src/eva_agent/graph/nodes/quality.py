"""Run the quality expert and record one dimension result."""

from eva_agent.budget.context import budget_context
from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.experts.quality import QualityExpert
from eva_agent.graph.state import ExpertState


async def quality_node(state: ExpertState, deps: GraphDependencies) -> dict:
    dimension_plan = state.get("dimension_plan")
    evidence = state.get("evidence")
    if dimension_plan is None or evidence is None:
        await deps.event_sink(
            state["run_id"],
            "quality.failed",
            {"error": "quality 专家缺少计划或证据"},
        )
        return {"status": RunStatus.FAILED, "errors": ["quality 专家缺少计划或证据"]}

    if dimension_plan.dimension is not Dimension.QUALITY:
        await deps.event_sink(
            state["run_id"],
            "quality.failed",
            {"error": "quality 专家收到错误的维度"},
        )
        return {"status": RunStatus.FAILED, "errors": ["quality 专家收到错误的维度"]}

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

    if result.status == "failed":
        await deps.event_sink(
            state["run_id"],
            "quality.failed",
            {"error": result.error, "summary": result.summary},
        )
    else:
        await deps.event_sink(
            state["run_id"],
            "quality.completed",
            {"status": result.status, "score": result.score},
        )

    if result.truncated_reason == "budget":
        context = budget_context()
        snapshot = context.ledger.snapshot() if context is not None else None
        await deps.event_sink(
            state["run_id"],
            "budget.exhausted",
            {
                "spent_usd": float(snapshot.spent_usd) if snapshot is not None else 0.0,
                "remaining_usd": float(snapshot.remaining_usd) if snapshot is not None else 0.0,
                "dimension": "quality",
            },
        )

    return {"expert_results": [result]}
