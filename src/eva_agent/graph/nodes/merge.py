"""Merge dimension results into one run-level result."""

from statistics import mean

from eva_agent.domain.result import MergedResult
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.state import EvalState


async def merge_node(state: EvalState, deps: GraphDependencies) -> dict:
    plan = state["plan"]
    results = list(state.get("expert_results") or [])
    skipped = list(plan.skipped) if plan is not None else []
    if not results:
        # An all-skipped plan has nothing to run; the run completes with its
        # skipped annotations. An enabled dimension that produced no result,
        # or a missing plan, is still a failure.
        all_skipped = plan is not None and not any(item.enabled for item in plan.dimensions)
        status = RunStatus.COMPLETED if all_skipped else RunStatus.FAILED
        metrics = {"overall_score": None} if all_skipped else {}
        merged = MergedResult(status=status, results=[], skipped=skipped, metrics=metrics)
        await deps.event_sink(state["run_id"], "run.merged", {"status": merged.status.value})
        return {"merged": merged, "status": merged.status}

    successful = [result for result in results if result.status in {"ok", "partial"} and result.score is not None]
    scores = [result.score for result in successful if result.score is not None]
    overall = mean(scores) if scores else None
    status = RunStatus.COMPLETED if successful else RunStatus.FAILED
    merged = MergedResult(
        status=status,
        results=results,
        skipped=skipped,
        metrics={"overall_score": overall},
    )
    await deps.event_sink(
        state["run_id"],
        "run.merged",
        {"status": status.value, "overall_score": overall},
    )
    return {"merged": merged, "status": status}
