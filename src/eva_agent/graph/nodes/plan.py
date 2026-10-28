"""Ask the planner for an evaluation plan and write the draft."""

from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.plan_validation import PlanValidationError
from eva_agent.graph.state import EvalState
from eva_agent.runtime.artifacts import run_paths, write_plan_draft


async def plan_node(state: EvalState, deps: GraphDependencies) -> dict:
    evidence = state["evidence"]
    if evidence is None:
        return {"status": RunStatus.FAILED, "errors": ["缺少证据，无法制定计划"]}

    try:
        plan = await deps.planner.plan(
            evidence,
            state["budget"],
            objective=state["input"].objective,
        )
    except PlanValidationError as exc:
        return {"status": RunStatus.FAILED, "errors": [f"评估计划无效: {exc}"]}
    except Exception as exc:
        return {"status": RunStatus.FAILED, "errors": [f"制定评估计划失败: {exc}"]}

    paths = run_paths(deps.data_dir, state["run_id"])
    try:
        write_plan_draft(paths, plan, overwrite=True)
    except Exception as exc:
        return {"status": RunStatus.FAILED, "errors": [f"写入计划草案失败: {exc}"]}

    await deps.event_sink(
        state["run_id"],
        "plan.ready",
        {"plan_path": str(paths.plan_draft)},
    )
    return {"plan": plan, "status": RunStatus.WAITING_APPROVAL}
