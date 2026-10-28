from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from eva_agent.adapters.openjudge.catalog import GraderEntry
from eva_agent.adapters.target.base import TargetTransport
from eva_agent.pricing import PricingTable
from eva_agent.security.target_policy import TargetPolicy
from eva_agent.tools.dataset import build_sample_dataset_tool
from eva_agent.tools.declarative import build_run_declarative_checks_tool
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.graders import GraderRunner, build_list_graders_tool, build_run_graders_tool
from eva_agent.tools.pricing import build_get_pricing_tool
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.rubric import RubricGenerator, build_generate_rubric_tool
from eva_agent.tools.target import build_call_target_tool, build_read_target_probe_tool
from eva_agent.tools.tokens import build_count_tokens_tool


@dataclass(frozen=True)
class ToolDependencies:
    catalog: dict[str, GraderEntry]
    runner: GraderRunner
    rubric_generator: RubricGenerator
    max_bytes: int
    max_rows: int
    policy: TargetPolicy | None = None
    transport: TargetTransport | None = None
    pricing: PricingTable | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    token_encoding: str = "cl100k_base"
    target_timeout_s: float = 30.0
    max_response_bytes: int = 1_048_576


def build_tool_registry(dependencies: ToolDependencies) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_read_evidence_tool(dependencies.max_bytes))
    registry.register(build_sample_dataset_tool(max_bytes=dependencies.max_bytes, max_rows=dependencies.max_rows))
    registry.register(build_list_graders_tool(dependencies.catalog))
    registry.register(build_generate_rubric_tool(dependencies.rubric_generator))
    registry.register(build_run_graders_tool(dependencies.runner, max_rows=dependencies.max_rows))
    registry.register(build_count_tokens_tool(dependencies.token_encoding))
    registry.register(build_read_target_probe_tool())
    registry.register(build_run_declarative_checks_tool(max_rows=dependencies.max_rows))
    if dependencies.policy is not None and dependencies.transport is not None:
        registry.register(
            build_call_target_tool(
                policy=dependencies.policy,
                transport=dependencies.transport,
                env=dependencies.env,
                default_max_response_bytes=dependencies.max_response_bytes,
                token_encoding=dependencies.token_encoding,
            )
        )
    if dependencies.pricing is not None:
        registry.register(build_get_pricing_tool(dependencies.pricing))
    return registry
