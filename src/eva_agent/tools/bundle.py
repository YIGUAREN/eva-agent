"""Assemble the phase-1 tool registry."""

from __future__ import annotations

from dataclasses import dataclass

from eva_agent.adapters.openjudge.catalog import GraderEntry
from eva_agent.tools.dataset import build_sample_dataset_tool
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.graders import GraderRunner, build_list_graders_tool, build_run_graders_tool
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.rubric import RubricGenerator, build_generate_rubric_tool


@dataclass(frozen=True)
class ToolDependencies:
    catalog: dict[str, GraderEntry]
    runner: GraderRunner
    rubric_generator: RubricGenerator
    max_bytes: int
    max_rows: int


def build_tool_registry(dependencies: ToolDependencies) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_read_evidence_tool(dependencies.max_bytes))
    registry.register(build_sample_dataset_tool(max_bytes=dependencies.max_bytes, max_rows=dependencies.max_rows))
    registry.register(build_list_graders_tool(dependencies.catalog))
    registry.register(build_generate_rubric_tool(dependencies.rubric_generator))
    registry.register(build_run_graders_tool(dependencies.runner, max_rows=dependencies.max_rows))
    return registry
