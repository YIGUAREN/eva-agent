from pathlib import Path

import pytest

from eva_agent.adapters.inputs.paste import write_paste_workspace
from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.result import (
    GraderRunOutput,
    GraderSampleOutcome,
    NormalizedQuality,
    Usage,
)
from eva_agent.domain.run import DatasetRef, InputKind, PastedInput
from eva_agent.graph.experts.quality import (
    QualityExpert,
    aggregate_outcomes,
    summarize_quality,
)
from eva_agent.graph.experts.rows import prepare_rows_with_indexes
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.graders import RunGradersInput
from eva_agent.tools.registry import ToolRegistry, ToolSpec

ROWS = [
    {"query": "q1", "response": "a1", "context": "", "reference": "r1"},
    {"query": "q2", "response": "a2", "context": "", "reference": "r2"},
]


def quality_spec() -> GraderSpec:
    return GraderSpec(level="builtin", name="relevance", builtin_id="relevance")


def dataset_evidence() -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.DATASET,
        root="/tmp/evidence",
        files=[],
        dataset=DatasetRef(path="dataset.csv", format="csv"),
        manifest_sha256="a" * 64,
    )


def upload_evidence(root: str) -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.UPLOAD,
        root=root,
        files=[EvidenceFile(path="qa.jsonl", size=1, sha256="a" * 64)],
        manifest_sha256="a" * 64,
    )


def quality_plan() -> DimensionPlan:
    return DimensionPlan(
        dimension=Dimension.QUALITY,
        enabled=True,
        sample_size=5,
        graders=[quality_spec()],
    )


def build_tools(outcomes: list[GraderSampleOutcome]) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_read_evidence_tool(1_000_000))

    async def run_handler(payload: RunGradersInput) -> GraderRunOutput:
        return GraderRunOutput(outcomes=outcomes, duration_ms=5)

    registry.register(
        ToolSpec(
            name="run_graders",
            description="fake runner",
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


def test_scores_are_aggregated_per_sample_and_overall() -> None:
    output = GraderRunOutput(
        outcomes=[
            GraderSampleOutcome(grader="relevance", sample_index=0, score=1.0),
            GraderSampleOutcome(grader="relevance", sample_index=1, score=0.0),
        ],
        duration_ms=7,
    )

    normalized = aggregate_outcomes(output, specs=[quality_spec()], rows=ROWS, evidence=dataset_evidence())

    assert normalized.score == 0.5
    assert normalized.confidence == 1.0
    assert normalized.metrics["scored_sample_count"] == 2
    assert normalized.metrics["scored_outcomes"] == 2
    assert normalized.usage.duration_ms == 7


def test_low_score_produces_one_finding_per_sample() -> None:
    output = GraderRunOutput(
        outcomes=[
            GraderSampleOutcome(grader="relevance", sample_index=0, score=1.0),
            GraderSampleOutcome(grader="relevance", sample_index=1, score=0.2),
        ]
    )

    normalized = aggregate_outcomes(output, specs=[quality_spec()], rows=ROWS, evidence=dataset_evidence())

    assert [finding.id for finding in normalized.findings] == ["quality-low-2"]
    assert normalized.findings[0].severity == "medium"
    assert normalized.findings[0].evidence[0].locator == "dataset.csv#1"


def test_errors_lower_confidence_but_keep_scores() -> None:
    output = GraderRunOutput(
        outcomes=[
            GraderSampleOutcome(grader="relevance", sample_index=0, score=1.0),
            GraderSampleOutcome(grader="relevance", sample_index=1, error="boom"),
        ]
    )

    normalized = aggregate_outcomes(output, specs=[quality_spec()], rows=ROWS, evidence=dataset_evidence())

    assert normalized.score == 1.0
    assert normalized.confidence == 0.5
    assert normalized.metrics["errors"] == ["boom"]


def test_all_failed_has_no_score() -> None:
    output = GraderRunOutput(
        outcomes=[
            GraderSampleOutcome(grader="relevance", sample_index=0, error="boom"),
            GraderSampleOutcome(grader="relevance", sample_index=1, error="boom"),
        ]
    )

    normalized = aggregate_outcomes(output, specs=[quality_spec()], rows=ROWS, evidence=dataset_evidence())

    assert normalized.score is None
    assert normalized.confidence == 0.0
    assert summarize_quality(normalized, sample_count=2, spec_count=1).status == "failed"


def test_summarize_ok_partial_failed() -> None:
    ok = summarize_quality(
        NormalizedQuality(
            score=0.8,
            confidence=1.0,
            metrics={"scored_sample_count": 2},
            usage=Usage(),
        ),
        sample_count=2,
        spec_count=1,
    )
    partial = summarize_quality(
        NormalizedQuality(
            score=0.8,
            confidence=0.5,
            metrics={"scored_sample_count": 1},
            usage=Usage(),
        ),
        sample_count=2,
        spec_count=1,
    )
    failed = summarize_quality(
        NormalizedQuality(score=None, confidence=0.0, metrics={}, usage=Usage()),
        sample_count=2,
        spec_count=1,
    )

    assert (ok.status, partial.status, failed.status) == ("ok", "partial", "failed")
    assert ok.summary.startswith("quality 完成")
    assert partial.summary.startswith("quality 部分成功")
    assert failed.error == "all grader calls failed"


async def test_expert_runs_paste_end_to_end(tmp_path: Path) -> None:
    evidence = write_paste_workspace("run-1", PastedInput(query="q", response="a"), tmp_path)
    tools = build_tools([GraderSampleOutcome(grader="relevance", sample_index=0, score=0.9)])
    expert = QualityExpert(tools, default_sample_size=5)

    result = await expert.run(run_id="run-1", plan=quality_plan(), evidence=evidence)

    assert result.status == "ok"
    assert result.score == 0.9
    assert result.dimension is Dimension.QUALITY
    assert result.evidence[0].locator == "paste.json"


async def test_expert_rejects_missing_response(tmp_path: Path) -> None:
    evidence = write_paste_workspace("run-1", PastedInput(query="q"), tmp_path)
    expert = QualityExpert(build_tools([]), default_sample_size=5)

    with pytest.raises(ValueError, match="no rows"):
        await expert.run(run_id="run-1", plan=quality_plan(), evidence=evidence)


async def test_upload_rows_are_parsed_from_qa_jsonl(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "qa.jsonl").write_text(
        '{"query": "q1", "response": "a1", "context": "ctx", "reference": "ref"}\n'
        "\n"
        '{"query": "q2", "response": "a2"}\n',
        encoding="utf-8",
    )

    rows, indexes = await prepare_rows_with_indexes(
        build_tools([]), upload_evidence(str(evidence_dir)), run_id="run-1", limit=10
    )

    assert rows == [
        {"query": "q1", "response": "a1", "context": "ctx", "reference": "ref"},
        {"query": "q2", "response": "a2", "context": "", "reference": ""},
    ]
    assert indexes == [None, None]


async def test_upload_rows_require_qa_jsonl(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "qa.txt").write_text("not jsonl", encoding="utf-8")

    with pytest.raises(ValueError, match="upload evidence has no qa.jsonl") as exc:
        await prepare_rows_with_indexes(
            build_tools([]), upload_evidence(str(evidence_dir)), run_id="run-1", limit=10
        )

    assert str(tmp_path) not in str(exc.value)


@pytest.mark.parametrize("line", ["not json", "[]", '{"query": "q"}'])
async def test_upload_rows_reject_malformed_lines(tmp_path: Path, line: str) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "qa.jsonl").write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="qa.jsonl is malformed"):
        await prepare_rows_with_indexes(
            build_tools([]), upload_evidence(str(evidence_dir)), run_id="run-1", limit=10
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"query": "q", "response": 4}',
        '{"query": null, "response": "a"}',
        '{"query": "q", "response": "a", "context": 1}',
        '{"query": "q", "response": "a", "reference": null}',
    ],
)
async def test_upload_rows_reject_non_string_fields(tmp_path: Path, payload: str) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "qa.jsonl").write_text(payload + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="qa.jsonl is malformed"):
        await prepare_rows_with_indexes(
            build_tools([]), upload_evidence(str(evidence_dir)), run_id="run-1", limit=10
        )


async def test_upload_rows_honor_the_row_limit(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "qa.jsonl").write_text(
        '{"query": "q1", "response": "a1"}\n'
        '{"query": "q2", "response": "a2"}\n'
        '{"query": "q3", "response": "a3"}\n',
        encoding="utf-8",
    )

    rows, indexes = await prepare_rows_with_indexes(
        build_tools([]), upload_evidence(str(evidence_dir)), run_id="run-1", limit=2
    )

    assert [row["query"] for row in rows] == ["q1", "q2"]
    assert indexes == [None, None]


def test_upload_refs_cite_qa_jsonl() -> None:
    rows = [{"query": "q", "response": "a", "context": "", "reference": ""}]
    output = GraderRunOutput(outcomes=[GraderSampleOutcome(grader="relevance", sample_index=0, score=0.2)])

    normalized = aggregate_outcomes(
        output, specs=[quality_spec()], rows=rows, evidence=upload_evidence("/tmp/evidence")
    )

    assert normalized.findings[0].evidence[0].kind == "upload_row"
    assert normalized.findings[0].evidence[0].locator == "qa.jsonl#0"
    assert normalized.evidence[0].locator == "qa.jsonl"
