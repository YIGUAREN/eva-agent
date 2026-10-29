"""Write the final JSON and Markdown report."""

from datetime import UTC, datetime

from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.state import EvalState
from eva_agent.report.render import write_report


async def report_node(state: EvalState, deps: GraphDependencies) -> dict:
    plan = state["plan"]
    merged = state["merged"]
    if plan is None or merged is None:
        return {"status": RunStatus.FAILED, "errors": ["缺少计划或合并结果，无法生成报告"]}

    try:
        report = write_report(
            data_dir=deps.data_dir,
            run_id=state["run_id"],
            plan=plan,
            merged=merged,
            probe=state.get("probe"),
            generated_at=datetime.now(UTC),
        )
    except Exception as exc:
        await deps.event_sink(state["run_id"], "report.failed", {"error": str(exc)})
        return {"status": RunStatus.FAILED, "errors": [f"生成报告失败: {exc}"]}

    await deps.event_sink(
        state["run_id"],
        "report.ready",
        {"markdown_path": report.markdown_path, "json_path": report.json_path},
    )
    return {"report": report, "status": merged.status}
