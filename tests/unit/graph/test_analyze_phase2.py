from typing import cast

from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import DimensionResult, MergedResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.nodes.analyze import analyze_node
from eva_agent.graph.state import EvalState
from tests.support.runtime import dependencies


def result(dimension: Dimension, score: float | None, metrics: dict) -> DimensionResult:
    return DimensionResult(
        dimension=dimension,
        status="ok" if score is not None else "failed",
        score=score,
        confidence=1.0,
        summary="ok",
        metrics=metrics,
        usage=Usage(),
    )


def state(merged: MergedResult) -> EvalState:
    return cast(
        EvalState,
        {
            "run_id": "run-1",
            "input": None,
            "budget": None,
            "evidence": None,
            "plan": None,
            "approval": None,
            "probe": None,
            "status": RunStatus.RUNNING,
            "errors": [],
            "expert_results": [],
            "merged": merged,
            "report": None,
        },
    )


async def test_analyze_adds_cost_per_quality_point(tmp_path) -> None:
    merged = MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            result(Dimension.QUALITY, 0.5, {}),
            result(Dimension.COST, 0.9, {"cost_total_usd": 0.25}),
        ],
    )

    update = await analyze_node(state(merged), dependencies(tmp_path))

    assert update["merged"].metrics["cost_per_quality_point"] == 0.5


async def test_analyze_skips_cross_metric_without_quality(tmp_path) -> None:
    merged = MergedResult(
        status=RunStatus.COMPLETED,
        results=[result(Dimension.COST, 0.9, {"cost_total_usd": 0.25})],
    )

    update = await analyze_node(state(merged), dependencies(tmp_path))

    assert "cost_per_quality_point" not in update["merged"].metrics


async def test_analyze_skips_zero_quality_score(tmp_path) -> None:
    merged = MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            result(Dimension.QUALITY, 0.0, {}),
            result(Dimension.COST, 0.9, {"cost_total_usd": 0.25}),
        ],
    )

    update = await analyze_node(state(merged), dependencies(tmp_path))

    assert "cost_per_quality_point" not in update["merged"].metrics
