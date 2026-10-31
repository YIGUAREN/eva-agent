"""Everything the graph needs from the outside world."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

from eva_agent.domain.target import TargetSpec
from eva_agent.graph.events import EventSink, noop_event_sink
from eva_agent.graph.planner import Planner
from eva_agent.security.audit import AuditSink, noop_audit_sink
from eva_agent.security.scan_policy import ScanPolicy
from eva_agent.security.target_policy import TargetPolicy
from eva_agent.tools.registry import ToolRegistry


class GateRow(Protocol):
    """The fields the gate needs from one persisted decision row."""

    seq: int
    payload_json: dict[str, Any]


class GateStore(Protocol):
    async def next_seq(self, run_id: str) -> int: ...

    async def record(self, run_id: str, seq: int, action: str, payload: dict[str, Any]) -> bool: ...

    async def latest_gate(self, run_id: str) -> GateRow | None: ...


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
    scan_policy: ScanPolicy | None = None
    audit: AuditSink = field(default=noop_audit_sink)
    aig_configured: bool = False
    finalize_estimator: Callable[[Mapping[str, Any]], Decimal] | None = None
    gate_store: GateStore | None = None
