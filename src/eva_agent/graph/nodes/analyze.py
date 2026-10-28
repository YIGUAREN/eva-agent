"""Compute run-level summary metrics from merged results."""

from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.state import EvalState


async def analyze_node(state: EvalState, deps: GraphDependencies) -> dict:
    merged = state["merged"]
    if merged is None:
        return {"status": RunStatus.FAILED, "errors": ["没有可分析的合并结果"]}

    findings = [finding for result in merged.results for finding in result.findings]
    high_severity = [item for item in findings if item.severity in {"high", "critical"}]
    dimensions = {result.dimension.value: {"status": result.status, "score": result.score} for result in merged.results}
    metrics = {
        **merged.metrics,
        "dimensions": dimensions,
        "finding_count": len(findings),
        "high_severity_count": len(high_severity),
        "failed_dimensions": [result.dimension.value for result in merged.results if result.status == "failed"],
    }
    analyzed = merged.model_copy(update={"metrics": metrics})
    await deps.event_sink(state["run_id"], "analysis.completed", {"metrics": metrics})
    return {"merged": analyzed}
