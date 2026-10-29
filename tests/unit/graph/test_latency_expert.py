from datetime import UTC, datetime
from pathlib import Path

import pytest

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.grader import CheckSpec, GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graph.experts.latency import LatencyExpert, nearest_rank
from eva_agent.runtime.probe import ProbeStore
from eva_agent.tools.declarative import build_run_declarative_checks_tool
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.target import build_read_target_probe_tool


def record(
    index: int,
    latency_ms: int,
    *,
    status_code: int | None = 200,
    error: str | None = None,
) -> ProbeRecord:
    return ProbeRecord(
        index=index,
        prompt="q",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        status_code=status_code,
        latency_ms=latency_ms,
        response_text="ok",
        error=error,
    )


def demo_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )


def probe_evidence(tmp_path: Path, records: list[ProbeRecord]) -> EvidenceBundle:
    run_root = tmp_path / "runs" / "run-1"
    store = ProbeStore(run_root / "target_probe.jsonl")
    store.reset()
    for item in records:
        store.append(item)
    return EvidenceBundle(
        kind=InputKind.TARGET,
        root=str(run_root / "evidence"),
        target=demo_target(),
        target_prompts=["q"],
        manifest_sha256="a" * 64,
    )


def latency_plan(*, sample_size: int = 3, threshold: float = 1000) -> DimensionPlan:
    return DimensionPlan(
        dimension=Dimension.LATENCY,
        enabled=True,
        sample_size=sample_size,
        graders=[
            GraderSpec(
                level="declarative",
                name="latency_budget",
                input_map={"latency_ms": "$.latency_ms", "response": "$.response"},
                checks=[
                    CheckSpec(
                        type="numeric_threshold",
                        field="latency_ms",
                        op="<=",
                        value=threshold,
                    )
                ],
            )
        ],
    )


def build_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_read_target_probe_tool())
    registry.register(build_run_declarative_checks_tool(max_rows=100))
    return registry


async def run_expert(
    tmp_path: Path,
    records: list[ProbeRecord],
    plan: DimensionPlan,
):
    evidence = probe_evidence(tmp_path, records)
    return await LatencyExpert(build_tools()).run(
        run_id="run-1",
        plan=plan,
        evidence=evidence,
    )


def test_nearest_rank_uses_ceiling_rank() -> None:
    values = [10, 20, 30, 40]

    assert nearest_rank(values, 50) == 20
    assert nearest_rank(values, 95) == 40
    assert nearest_rank(values, 99) == 40
    assert nearest_rank([], 50) is None


async def test_all_successful_is_ok(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0, 10), record(1, 20), record(2, 30), record(3, 40)],
        latency_plan(sample_size=4),
    )

    assert result.dimension is Dimension.LATENCY
    assert result.status == "ok"
    assert result.score == 1.0
    assert result.confidence == 1.0
    assert result.metrics["count"] == 4
    assert result.metrics["success_count"] == 4
    assert result.metrics["failure_count"] == 0
    assert result.metrics["p50_ms"] == 20
    assert result.metrics["p95_ms"] == 40
    assert result.metrics["p99_ms"] == 40
    assert result.metrics["mean_ms"] == 25.0
    assert result.metrics["min_ms"] == 10
    assert result.metrics["max_ms"] == 40
    assert result.metrics["first_token_ms"] is None
    assert result.metrics["first_token_note"] == "non-streaming target"
    assert result.findings == []
    assert result.usage.samples == 4
    assert result.usage.grader_calls == 4
    assert result.evidence[0].locator == "target_probe.jsonl"


async def test_failure_makes_the_dimension_partial(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [
            record(0, 10),
            record(1, 120, status_code=500, error="http 500"),
            record(2, 30),
        ],
        latency_plan(sample_size=3),
    )

    assert result.status == "partial"
    assert result.score == 1.0
    assert result.confidence == pytest.approx(2 / 3)
    assert result.metrics["error_rate"] == 0.3333
    assert result.metrics["status_codes"] == {"200": 2, "500": 1}
    assert result.metrics["p95_ms"] == 30
    assert result.findings[0].id == "latency-errors"
    assert result.findings[0].severity == "medium"
    assert "http 500" in result.findings[0].description
    assert result.findings[0].evidence[0].locator == "target_probe.jsonl#1"


async def test_all_failed_has_no_score(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [
            record(0, 100, status_code=500, error="http 500"),
            record(1, 100, status_code=None, error="timeout"),
        ],
        latency_plan(sample_size=2),
    )

    assert result.status == "failed"
    assert result.score is None
    assert result.confidence == 0.0
    assert result.error == "no successful target requests"
    assert result.metrics["p95_ms"] is None
    assert result.metrics["status_codes"] == {"500": 1, "none": 1}
    assert result.findings[0].id == "latency-errors"


async def test_declarative_score_drives_the_dimension_score(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0, 10), record(1, 20)],
        latency_plan(sample_size=2, threshold=15),
    )

    assert result.score == 0.5
    assert result.metrics["grader_scores"] == {"latency_budget": 0.5}


async def test_all_successful_samples_failing_checks_fail_the_dimension(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0, 100), record(1, 200)],
        latency_plan(sample_size=2, threshold=50),
    )

    assert result.status == "failed"
    assert result.score == 0.0
    assert result.confidence == 1.0
    assert result.error == "all declarative checks failed"
    assert result.metrics["grader_scores"] == {"latency_budget": 0.0}
    assert result.metrics["errors"] == ["numeric_threshold failed"]


async def test_p95_above_the_l2_threshold_adds_a_low_finding(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0, 10), record(1, 20)],
        latency_plan(sample_size=2, threshold=15),
    )

    assert [finding.id for finding in result.findings] == ["latency-p95"]
    assert result.findings[0].severity == "low"
    assert "20" in result.findings[0].description
    assert result.findings[0].evidence[0].locator == "target_probe.jsonl#1"


async def test_sample_size_limits_the_records(tmp_path: Path) -> None:
    result = await run_expert(
        tmp_path,
        [record(0, 10), record(1, 20), record(2, 30)],
        latency_plan(sample_size=2),
    )

    assert result.metrics["count"] == 2
    assert result.metrics["p95_ms"] == 20
    assert result.usage.samples == 2
    assert result.status == "ok"


async def test_expert_rejects_non_target_evidence(tmp_path: Path) -> None:
    evidence = EvidenceBundle(
        kind=InputKind.PASTE,
        pasted={"query": "q", "response": "a"},
        manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="requires target evidence"):
        await LatencyExpert(build_tools()).run(
            run_id="run-1",
            plan=latency_plan(),
            evidence=evidence,
        )
