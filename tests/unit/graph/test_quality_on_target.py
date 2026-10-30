from datetime import UTC, datetime
from pathlib import Path

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.experts.quality import QualityExpert
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.probe import ProbeStore
from eva_agent.tools.graders import RunGradersInput
from eva_agent.tools.registry import ToolRegistry, ToolSpec
from eva_agent.tools.target import build_read_target_probe_tool


def evidence(tmp_path: Path) -> EvidenceBundle:
    target = TargetSpec(
        name="demo", kind=TargetKind.OPENAI_COMPATIBLE, provider="demo", model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )
    return EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(run_paths(tmp_path / "data", "run-1").evidence),
        target=target,
        target_prompts=["q1", "q2"],
        manifest_sha256="a" * 64,
    )


def write_probe(tmp_path: Path) -> None:
    paths = run_paths(tmp_path / "data", "run-1")
    paths.root.mkdir(parents=True, exist_ok=True)
    store = ProbeStore(paths.target_probe)
    store.reset()
    store.append(
        ProbeRecord(
            index=0, prompt="q1", started_at=datetime.now(UTC), status_code=200,
            latency_ms=10, response_text="a1",
        )
    )
    store.append(
        ProbeRecord(
            index=1, prompt="q2", started_at=datetime.now(UTC), status_code=None,
            latency_ms=10, response_text="", error="timeout",
        )
    )


async def test_quality_expert_cites_source_probe_index(tmp_path: Path) -> None:
    paths = run_paths(tmp_path / "data", "run-1")
    paths.root.mkdir(parents=True, exist_ok=True)
    store = ProbeStore(paths.target_probe)
    store.reset()
    store.append(
        ProbeRecord(
            index=0, prompt="q1", started_at=datetime.now(UTC), status_code=None,
            latency_ms=10, response_text="", error="timeout",
        )
    )
    store.append(
        ProbeRecord(
            index=1, prompt="q2", started_at=datetime.now(UTC), status_code=200,
            latency_ms=10, response_text="a2",
        )
    )

    async def handler(payload: RunGradersInput) -> GraderRunOutput:
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(grader="relevance", sample_index=index, score=0.2)
                for index in range(len(payload.rows))
            ]
        )

    tools = ToolRegistry()
    tools.register(build_read_target_probe_tool())
    tools.register(
        ToolSpec(
            name="run_graders", description="fake", input_model=RunGradersInput,
            output_model=GraderRunOutput, permission="internal", timeout_s=10,
            side_effects=False, idempotent=False, handler=handler,
        )
    )
    plan = DimensionPlan(
        dimension=Dimension.QUALITY, enabled=True, sample_size=2,
        graders=[GraderSpec(level="builtin", name="relevance", builtin_id="relevance")],
    )
    result = await QualityExpert(tools, default_sample_size=5).run(
        run_id="run-1", plan=plan, evidence=evidence(tmp_path)
    )

    assert len(result.findings) == 1
    assert result.findings[0].evidence[0].locator == "target_probe.jsonl#1"


async def test_quality_expert_grades_successful_target_rows(tmp_path: Path) -> None:
    write_probe(tmp_path)
    seen: list[RunGradersInput] = []

    async def handler(payload: RunGradersInput) -> GraderRunOutput:
        seen.append(payload)
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(grader="relevance", sample_index=index, score=0.8)
                for index in range(len(payload.rows))
            ]
        )

    tools = ToolRegistry()
    tools.register(build_read_target_probe_tool())
    tools.register(
        ToolSpec(
            name="run_graders", description="fake", input_model=RunGradersInput,
            output_model=GraderRunOutput, permission="internal", timeout_s=10,
            side_effects=False, idempotent=False, handler=handler,
        )
    )
    plan = DimensionPlan(
        dimension=Dimension.QUALITY, enabled=True, sample_size=2,
        graders=[GraderSpec(level="builtin", name="relevance", builtin_id="relevance")],
    )
    result = await QualityExpert(tools, default_sample_size=5).run(
        run_id="run-1", plan=plan, evidence=evidence(tmp_path)
    )
    assert result.status == "ok"
    assert result.score == 0.8
    assert seen[0].rows == [
        {"query": "q1", "response": "a1", "context": "", "reference": ""}
    ]
    assert result.findings == []
