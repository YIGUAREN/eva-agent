"""Build a dimension node around a single expert."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.result import DimensionResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.state import ExpertState
from eva_agent.tools.registry import ToolRegistry


class ExpertRunner(Protocol):
    async def run(
        self,
        *,
        run_id: str,
        plan: DimensionPlan,
        evidence: EvidenceBundle,
    ) -> DimensionResult: ...


ExpertFactory = Callable[[ToolRegistry], ExpertRunner]


def make_expert_node(
    *,
    name: str,
    dimension: Dimension,
    expert_factory: ExpertFactory,
) -> Callable[[ExpertState, GraphDependencies], Awaitable[dict]]:
    """Return a node that validates state, runs one expert, and emits its events."""

    async def node(state: ExpertState, deps: GraphDependencies) -> dict:
        dimension_plan = state.get("dimension_plan")
        evidence = state.get("evidence")
        if dimension_plan is None or evidence is None:
            message = f"{name} 专家缺少计划或证据"
            await deps.event_sink(state["run_id"], f"{name}.failed", {"error": message})
            return {"status": RunStatus.FAILED, "errors": [message]}

        if dimension_plan.dimension is not dimension:
            message = f"{name} 专家收到错误的维度"
            await deps.event_sink(state["run_id"], f"{name}.failed", {"error": message})
            return {"status": RunStatus.FAILED, "errors": [message]}

        if state.get("probe") is None:
            message = f"{name} 专家缺少 probe 摘要"
            await deps.event_sink(state["run_id"], f"{name}.failed", {"error": message})
            return {"status": RunStatus.FAILED, "errors": [message]}

        expert = expert_factory(deps.tools)
        try:
            result = await expert.run(
                run_id=state["run_id"],
                plan=dimension_plan,
                evidence=evidence,
            )
        except Exception as exc:
            result = DimensionResult(
                dimension=dimension,
                status="failed",
                score=None,
                confidence=0.0,
                summary=f"{name} 专家执行失败",
                usage=Usage(),
                error=str(exc),
            )

        if result.status == "failed":
            await deps.event_sink(
                state["run_id"],
                f"{name}.failed",
                {"error": result.error, "summary": result.summary},
            )
        else:
            await deps.event_sink(
                state["run_id"],
                f"{name}.completed",
                {"status": result.status, "score": result.score},
            )

        return {"expert_results": [result]}

    return node
