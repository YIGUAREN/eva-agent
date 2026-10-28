"""Everything the graph needs from the outside world."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from eva_agent.domain.target import TargetSpec
from eva_agent.graph.events import EventSink, noop_event_sink
from eva_agent.graph.planner import Planner
from eva_agent.security.target_policy import TargetPolicy
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
    targets: dict[str, TargetSpec] = field(default_factory=dict)
    policy: TargetPolicy | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    target_timeout_s: float = 30.0
    max_response_bytes: int = 1_048_576
    token_encoding: str = "cl100k_base"
