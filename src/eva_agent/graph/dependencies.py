"""Everything the graph needs from the outside world."""

from dataclasses import dataclass, field

from eva_agent.graph.events import EventSink, noop_event_sink
from eva_agent.graph.planner import Planner
from eva_agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class GraphDependencies:
    planner: Planner
    tools: ToolRegistry
    catalog_ids: frozenset[str]
    data_dir: str
    max_input_bytes: int
    max_dataset_rows: int
    default_sample_size: int = 50
    event_sink: EventSink = field(default=noop_event_sink)
