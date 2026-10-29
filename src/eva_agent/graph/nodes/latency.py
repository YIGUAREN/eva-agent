"""Run the latency expert and record one dimension result."""

from eva_agent.domain.plan import Dimension
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.experts.latency import LatencyExpert
from eva_agent.graph.nodes.expert import make_expert_node
from eva_agent.graph.state import ExpertState

_run = make_expert_node(
    name="latency",
    dimension=Dimension.LATENCY,
    expert_factory=LatencyExpert,
)


async def latency_node(state: ExpertState, deps: GraphDependencies) -> dict:
    return await _run(state, deps)
