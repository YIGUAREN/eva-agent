"""Write the final JSON and Markdown report."""

from datetime import UTC, datetime

from eva_agent.budget.context import budget_context
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
        # Usage provenance is computed inside the try so a malformed snapshot
        # fails the report event instead of escaping as an unhandled error.
        context = budget_context()
        snapshot = context.ledger.snapshot() if context is not None else None
        usage = {
            "estimated": snapshot.estimated if snapshot is not None else False,
            "cached": sum(result.usage.cached for result in merged.results),
            "fresh": sum(result.usage.fresh for result in merged.results),
            "budget_enforced": context.ledger.enforced if context is not None else False,
            "spent_usd": snapshot.spent_usd if snapshot is not None else 0,
        }
        report = write_report(
            data_dir=deps.data_dir,
            run_id=state["run_id"],
            plan=plan,
            merged=merged,
            probe=state.get("probe"),
            generated_at=datetime.now(UTC),
            usage=usage,
            finalize_overrun=bool(state.get("finalize_overrun", False)),
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
