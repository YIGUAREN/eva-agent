from typing import Literal

from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.nodes.merge import merge_node
from eva_agent.graph.state import EvalState
from tests.support.runtime import budget, dependencies, initial_state, paste_input


def security_only_plan() -> EvalPlan:
    return EvalPlan(
        objective="评估安全",
        evidence_summary="测试证据",
        dimensions=[DimensionPlan(dimension=Dimension.SECURITY, enabled=True)],
        skipped=[
            SkippedDimension(dimension=dimension, reason="测试跳过")
            for dimension in (Dimension.QUALITY, Dimension.LATENCY, Dimension.COST)
        ],
        budget=budget(),
    )


def result(
    status: Literal["ok", "partial", "skipped", "failed"], *, score: float | None = None
) -> DimensionResult:
    return DimensionResult(
        dimension=Dimension.SECURITY,
        status=status,
        score=score,
        confidence=0.0,
        summary=status,
        usage=Usage(),
    )


def state_with(*results: DimensionResult) -> EvalState:
    state = initial_state(paste_input())
    state["plan"] = security_only_plan()
    state["expert_results"] = list(results)
    return state


async def test_a_run_with_only_skipped_dimensions_completes(tmp_path) -> None:
    update = await merge_node(state_with(result("skipped")), dependencies(tmp_path))

    assert update["status"] is RunStatus.COMPLETED
    assert update["merged"].status is RunStatus.COMPLETED
    assert update["merged"].metrics == {"overall_score": None}


async def test_a_failed_dimension_still_fails_the_run(tmp_path) -> None:
    update = await merge_node(state_with(result("failed")), dependencies(tmp_path))

    assert update["status"] is RunStatus.FAILED


async def test_an_ok_dimension_scores_the_run(tmp_path) -> None:
    update = await merge_node(state_with(result("ok", score=0.8)), dependencies(tmp_path))

    assert update["status"] is RunStatus.COMPLETED
    assert update["merged"].metrics == {"overall_score": 0.8}
