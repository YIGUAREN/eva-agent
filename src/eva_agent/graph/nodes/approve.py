"""Pause for human approval, then freeze the approved plan."""

from datetime import UTC, datetime

from langgraph.types import interrupt

from eva_agent.domain.result import Approval
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.plan_validation import PlanValidationError, parse_plan_edit
from eva_agent.graph.state import EvalState
from eva_agent.runtime.artifacts import run_paths, write_plan_approved


async def approve_node(state: EvalState, deps: GraphDependencies) -> dict:
    plan = state["plan"]
    run_id = state["run_id"]
    if plan is None:
        return {"status": RunStatus.FAILED, "errors": ["没有计划可供审批"]}

    decision = interrupt({"run_id": run_id, "plan": plan.model_dump(mode="json")})
    if not isinstance(decision, dict) or decision.get("approved") is not True:
        await deps.event_sink(run_id, "plan.rejected", {})
        return {"status": RunStatus.CANCELLED}

    approved_plan = plan
    edited = decision.get("plan")
    if edited is not None:
        try:
            approved_plan = parse_plan_edit(edited, catalog_ids=set(deps.catalog_ids), original=plan)
        except PlanValidationError as exc:
            await deps.event_sink(run_id, "plan.rejected", {"error": str(exc)})
            return {"status": RunStatus.FAILED, "errors": [f"计划修改无效: {exc}"]}

    approval = Approval(plan=approved_plan, decided_at=datetime.now(UTC))
    paths = run_paths(deps.data_dir, run_id)
    try:
        write_plan_approved(paths, approval)
    except Exception as exc:
        await deps.event_sink(run_id, "run.failed", {"error": str(exc)})
        return {"status": RunStatus.FAILED, "errors": [f"写入批准计划失败: {exc}"]}

    await deps.event_sink(run_id, "plan.approved", {"plan_path": str(paths.plan_approved)})
    return {
        "approval": approval,
        "plan": approved_plan,
        "budget": approved_plan.budget,
        "status": RunStatus.RUNNING,
    }
