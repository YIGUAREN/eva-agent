import json
from datetime import UTC, datetime

import pytest

from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.result import (
    DimensionResult,
    EvidenceRef,
    Finding,
    MergedResult,
    Usage,
)
from eva_agent.domain.run import RunStatus
from eva_agent.report.render import build_report_payload, render_markdown, write_report


def plan() -> EvalPlan:
    return EvalPlan(
        objective="评估回答质量",
        evidence_summary="2 条样本",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.QUALITY,
                enabled=True,
                graders=[GraderSpec(level="builtin", name="relevance", builtin_id="relevance")],
            )
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.LATENCY, reason="阶段 1 未实现"),
            SkippedDimension(dimension=Dimension.COST, reason="阶段 1 未实现"),
            SkippedDimension(dimension=Dimension.SECURITY, reason="阶段 1 未实现"),
        ],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1),
    )


def merged(status: RunStatus = RunStatus.COMPLETED) -> MergedResult:
    return MergedResult(
        status=status,
        results=[
            DimensionResult(
                dimension=Dimension.QUALITY,
                status="ok",
                score=0.75,
                confidence=1.0,
                summary="quality 完成：2 个样本，平均 0.750",
                findings=[
                    Finding(
                        id="quality-low-2",
                        severity="medium",
                        category="quality",
                        title="样本 2 质量得分偏低",
                        description="平均质量得分 0.5",
                        evidence=[EvidenceRef(kind="dataset_row", locator="dataset.csv#1")],
                    )
                ],
                evidence=[EvidenceRef(kind="file", locator="dataset.csv")],
                usage=Usage(grader_calls=2, samples=2, duration_ms=30),
            )
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.LATENCY, reason="阶段 1 未实现"),
            SkippedDimension(dimension=Dimension.COST, reason="阶段 1 未实现"),
            SkippedDimension(dimension=Dimension.SECURITY, reason="阶段 1 未实现"),
        ],
        metrics={"overall_score": 0.75},
    )


def test_payload_is_json_safe_and_ordered() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        generated_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
    )

    assert payload["run_id"] == "run-1"
    assert payload["status"] == "completed"
    assert payload["overall_score"] == 0.75
    assert payload["dimensions"][0]["dimension"] == "quality"
    assert payload["findings"][0]["id"] == "quality-low-2"


def test_markdown_has_required_sections() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        generated_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload)

    for heading in ["## 概览", "## 计划", "## 维度结果", "## 发现", "## 未执行维度", "## 限制"]:
        assert heading in markdown
    assert "### 质量" in markdown
    assert "样本 2 质量得分偏低" in markdown


def test_markdown_uses_chinese_labels_for_status_dimensions_and_severity() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        generated_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload)

    assert "- 状态：已完成" in markdown
    assert "### 质量" in markdown
    assert "- 状态：正常" in markdown
    assert "| quality-low-2 | 中 | 样本 2 质量得分偏低 |" in markdown
    assert "| 延迟 | 阶段 1 未实现 |" in markdown
    # The technical code values stay in the JSON payload, not the prose.
    assert "- 状态：`completed`" not in markdown


def test_markdown_handles_failed_dimension() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged(RunStatus.FAILED),
        generated_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload)

    assert "- 状态：失败" in markdown


def test_payload_is_json_serializable_with_non_json_metrics() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged().model_copy(
            update={"metrics": {"at": datetime(2026, 10, 27, 12, 0, tzinfo=UTC), "tags": {"a"}}}
        ),
        generated_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
    )

    text = json.dumps(payload, ensure_ascii=False)

    assert "2026-10-27T12:00:00" in text
    assert '"tags": ["a"]' in text


def test_markdown_escapes_table_cells() -> None:
    finding = Finding(
        id="quality-low-2",
        severity="medium",
        category="quality",
        title="a|b",
        description="d",
        evidence=[EvidenceRef(kind="dataset_row", locator="x|y")],
    )
    result = merged().results[0].model_copy(update={"findings": [finding]})
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged().model_copy(update={"results": [result]}),
        generated_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload)

    assert "a\\|b" in markdown
    assert "x\\|y" in markdown


def test_markdown_handles_empty_sections() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=MergedResult(status=RunStatus.COMPLETED),
        generated_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload)

    assert "（没有任何维度产生结果）" in markdown
    assert "（未发现问题）" in markdown
    assert "（无）" in markdown


def test_write_report_requires_aware_generated_at(tmp_path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        write_report(
            data_dir=tmp_path,
            run_id="run-1",
            plan=plan(),
            merged=merged(),
            generated_at=datetime(2026, 10, 27, 12, 0),
        )


def test_write_report_writes_both_files(tmp_path) -> None:
    report = write_report(
        data_dir=tmp_path,
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        generated_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC),
    )

    assert report.markdown_path.endswith("report.md")
    assert report.json_path.endswith("report.json")
    assert (tmp_path / "runs" / "run-1" / "report.md").is_file()
    assert (tmp_path / "runs" / "run-1" / "report.json").is_file()
