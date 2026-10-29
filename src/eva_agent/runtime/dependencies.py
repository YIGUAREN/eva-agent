"""Build production graph dependencies from application config."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from eva_agent.adapters.openjudge.catalog import load_catalog
from eva_agent.adapters.openjudge.models import build_openjudge_model
from eva_agent.adapters.openjudge.rubric import OpenJudgeRubricGenerator
from eva_agent.adapters.openjudge.runner import OpenJudgeRunner
from eva_agent.adapters.target.http import HttpCoreTransport
from eva_agent.config.settings import AppConfig
from eva_agent.config.targets import load_targets
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.planner import LlmPlanner
from eva_agent.llm.factory import build_agent_chat_model, structured_output_method
from eva_agent.pricing import PricingTable
from eva_agent.security.target_policy import TargetPolicy, load_security_policy
from eva_agent.tools.bundle import ToolDependencies, build_tool_registry

QUALITY_DEFAULT_SAMPLE_SIZE = 50


def build_runtime_dependencies(
    config: AppConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> GraphDependencies:
    environment = os.environ if env is None else env
    targets = load_targets(Path(config.target.targets_path))
    policy = TargetPolicy(load_security_policy(Path(config.security.policy_path)))
    pricing = PricingTable.load(Path(config.pricing.path))

    catalog = load_catalog(Path(config.openjudge.catalog_path))
    judge_model = build_openjudge_model(config.models.judge, environment)
    runner = OpenJudgeRunner(
        judge_model,
        catalog,
        max_concurrency=config.openjudge.max_concurrency,
        timeout_s=config.openjudge.grader_timeout_s,
    )
    rubric_generator = OpenJudgeRubricGenerator(judge_model)
    tools = build_tool_registry(
        ToolDependencies(
            catalog=catalog,
            runner=runner,
            rubric_generator=rubric_generator,
            max_bytes=config.pipeline.dataset_max_bytes,
            max_rows=config.pipeline.dataset_max_rows,
            policy=policy,
            transport=HttpCoreTransport(),
            pricing=pricing,
            env=environment,
            token_encoding=config.target.token_encoding,
            target_timeout_s=config.target.timeout_s,
            max_response_bytes=config.target.max_response_bytes,
        )
    )
    planner = LlmPlanner(
        build_agent_chat_model(config.models.agent, environment),
        tools,
        structured_method=structured_output_method(config.models.agent),
        sample_rows=config.pipeline.evidence_sample_rows,
    )
    return GraphDependencies(
        planner=planner,
        tools=tools,
        catalog_ids=frozenset(catalog),
        data_dir=config.storage.data_dir,
        max_input_bytes=config.pipeline.dataset_max_bytes,
        max_dataset_rows=config.pipeline.dataset_max_rows,
        default_sample_size=QUALITY_DEFAULT_SAMPLE_SIZE,
        targets=targets,
        policy=policy,
        env=environment,
        target_timeout_s=config.target.timeout_s,
        max_response_bytes=config.target.max_response_bytes,
        token_encoding=config.target.token_encoding,
    )
