"""Ask the planner for an evaluation plan and write the draft."""

from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.plan_validation import PlanValidationError
from eva_agent.graph.state import EvalState
from eva_agent.runtime.artifacts import run_paths, write_plan_draft


async def _fail(deps: GraphDependencies, run_id: str, message: str, *, error: str | None = None) -> dict:
    await deps.event_sink(run_id, "plan.failed", {"error": error or message})
    return {"status": RunStatus.FAILED, "errors": [message]}


async def plan_node(state: EvalState, deps: GraphDependencies) -> dict:
    evidence = state["evidence"]
    run_id = state["run_id"]
    if evidence is None:
        return await _fail(deps, run_id, "缺少证据，无法制定计划")

    try:
        plan = await deps.planner.plan(
            evidence,
            state["budget"],
            objective=state["input"].objective,
        )
    except PlanValidationError as exc:
        return await _fail(deps, run_id, f"评估计划无效: {exc}", error=str(exc))
    except Exception as exc:
        return await _fail(deps, run_id, f"制定评估计划失败: {exc}", error=str(exc))

    paths = run_paths(deps.data_dir, run_id)
    try:
        write_plan_draft(paths, plan, overwrite=True)
    except Exception as exc:
        return await _fail(deps, run_id, f"写入计划草案失败: {exc}", error=str(exc))

    await deps.event_sink(
        state["run_id"],
        "plan.ready",
        {"plan_path": str(paths.plan_draft)},
    )
    return {"plan": plan, "status": RunStatus.WAITING_APPROVAL}
