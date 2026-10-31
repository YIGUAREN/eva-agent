"""Shared fakes for phase-1 graph and manager tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from eva_agent.config.settings import (
    AgentSettings,
    AppConfig,
    ModelConfig,
    ModelsSettings,
    StorageSettings,
)
from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.domain.run import DatasetRef, InputKind, PastedInput, RunInput, RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.events import noop_event_sink
from eva_agent.graph.state import EvalState
from eva_agent.tools.dataset import build_sample_dataset_tool
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.graders import RunGradersInput
from eva_agent.tools.registry import ToolRegistry, ToolSpec


def budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1)


def paste_input(query: str = "q", response: str | None = "a") -> RunInput:
    return RunInput(
        kind=InputKind.PASTE,
        paste=PastedInput(query=query, response=response),
    )


def dataset_input(path: Path, *, format: Literal["csv", "jsonl"] = "csv") -> RunInput:
    return RunInput(
        kind=InputKind.DATASET,
        dataset=DatasetRef(path=str(path), format=format),
    )


def skipped_dimensions() -> list[SkippedDimension]:
    return [
        SkippedDimension(dimension=Dimension.LATENCY, reason="阶段 1 未实现"),
        SkippedDimension(dimension=Dimension.COST, reason="阶段 1 未实现"),
        SkippedDimension(dimension=Dimension.SECURITY, reason="阶段 1 未实现"),
    ]


def sample_plan(*, objective: str = "评估回答质量", sample_size: int = 5) -> EvalPlan:
    return EvalPlan(
        objective=objective,
        evidence_summary="测试证据",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.QUALITY,
                enabled=True,
                sample_size=sample_size,
                graders=[
                    GraderSpec(level="builtin", name="relevance", builtin_id="relevance"),
                    GraderSpec(level="llm_rubric", name="quality_rubric", rubric="- 回答直接解决问题"),
                ],
            )
        ],
        skipped=skipped_dimensions(),
        budget=budget(),
    )


class FakePlanner:
    def __init__(self, *, plan: EvalPlan | None = None, error: Exception | None = None) -> None:
        self._plan = plan or sample_plan()
        self._error = error
        self.calls = 0

    async def plan(
        self,
        evidence: object,
        budget: Budget,
        objective: str | None = None,
    ) -> EvalPlan:
        self.calls += 1
        if self._error is not None:
            raise self._error
        plan = self._plan.model_copy(update={"budget": budget})
        if objective:
            plan = plan.model_copy(update={"objective": objective})
        return plan


class FakeRunner:
    def __init__(
        self,
        *,
        score: float = 0.8,
        error: str | None = None,
        unscored: bool = False,
    ) -> None:
        self._score = score
        self._error = error
        self._unscored = unscored
        self.calls = 0

    async def run(self, specs: list[GraderSpec], rows: list[dict[str, Any]]) -> GraderRunOutput:
        self.calls += 1
        if self._error is not None:
            raise RuntimeError(self._error)
        if self._unscored:
            return GraderRunOutput(
                outcomes=[
                    GraderSampleOutcome(
                        grader=spec.name,
                        sample_index=index,
                        score=None,
                        error="grader unavailable",
                    )
                    for spec in specs
                    for index in range(len(rows))
                ],
                duration_ms=5,
            )
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(grader=spec.name, sample_index=index, score=self._score)
                for spec in specs
                for index in range(len(rows))
            ],
            duration_ms=5,
        )


def build_tools(runner: FakeRunner) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_read_evidence_tool(1_000_000))
    registry.register(build_sample_dataset_tool(max_bytes=1_000_000, max_rows=100))

    async def run_handler(payload: RunGradersInput) -> GraderRunOutput:
        return await runner.run(payload.specs, payload.rows)

    registry.register(
        ToolSpec(
            name="run_graders",
            description="fake OpenJudge runner",
            input_model=RunGradersInput,
            output_model=GraderRunOutput,
            permission="internal",
            timeout_s=10.0,
            side_effects=False,
            idempotent=False,
            handler=run_handler,
        )
    )
    return registry


def dependencies(
    tmp_path: Path,
    *,
    planner: FakePlanner | None = None,
    runner: FakeRunner | None = None,
    event_sink=noop_event_sink,
) -> GraphDependencies:
    runner = runner or FakeRunner()
    planner = planner or FakePlanner()
    return GraphDependencies(
        planner=planner,
        tools=build_tools(runner),
        catalog_ids=frozenset({"relevance"}),
        data_dir=str(tmp_path / "data"),
        max_input_bytes=1_000_000,
        max_dataset_rows=100,
        default_sample_size=5,
        event_sink=event_sink,
    )


def fake_config(tmp_path: Path) -> AppConfig:
    def model() -> ModelConfig:
        return ModelConfig(
            provider="openai-compatible",
            model="fake",
            base_url="http://localhost:1",
            api_key_env="FAKE_API_KEY",
        )

    return AppConfig(
        public_base_url="http://localhost",
        retention_days=30,
        models=ModelsSettings(agent=model(), judge=model(), attack=model()),
        agent=AgentSettings(max_steps=5),
        storage=StorageSettings(
            url=f"sqlite+aiosqlite:///{(tmp_path / 'eva.db').as_posix()}",
            data_dir=str(tmp_path / "data"),
            checkpoint_path=str(tmp_path / "checkpoints.sqlite"),
        ),
    )


def initial_state(run_input: RunInput, *, run_id: str = "run-1") -> EvalState:
    return {
        "run_id": run_id,
        "input": run_input,
        "budget": budget(),
        "evidence": None,
        "plan": None,
        "probe": None,
        "status": RunStatus.QUEUED,
        "errors": [],
        "approval": None,
        "expert_results": [],
        "merged": None,
        "report": None,
    }
