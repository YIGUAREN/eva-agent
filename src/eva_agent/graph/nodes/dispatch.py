"""Select the experts to run for an approved plan."""

from eva_agent.domain.plan import Dimension
from eva_agent.domain.run import RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.state import EvalState

SUPPORTED_DIMENSIONS = frozenset({Dimension.QUALITY, Dimension.LATENCY, Dimension.COST})


async def dispatch_node(state: EvalState, deps: GraphDependencies) -> dict:
    plan = state["plan"]
    if plan is None:
        return {"status": RunStatus.FAILED, "errors": ["没有计划可供调度"]}

    enabled = [
        item.dimension.value for item in plan.dimensions if item.enabled and item.dimension in SUPPORTED_DIMENSIONS
    ]
    if not enabled:
        return {"status": RunStatus.FAILED, "errors": ["没有可执行的维度"]}

    await deps.event_sink(state["run_id"], "dispatch.ready", {"dimensions": enabled})
    return {"status": RunStatus.RUNNING}
