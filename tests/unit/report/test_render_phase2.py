from datetime import UTC, datetime
from pathlib import Path

from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.probe import ProbeRecord, ProbeSummary
from eva_agent.domain.result import DimensionResult, MergedResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.report.render import build_report_payload, render_markdown, write_report
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.probe import ProbeStore


def plan() -> EvalPlan:
    return EvalPlan(
        objective="评估 demo target",
        evidence_summary="2 条 prompt",
        dimensions=[
            DimensionPlan(dimension=Dimension.LATENCY, enabled=True, sample_size=2),
            DimensionPlan(dimension=Dimension.COST, enabled=True, sample_size=2),
        ],
        skipped=[SkippedDimension(dimension=Dimension.SECURITY, reason="阶段 3 实现")],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
    )


def merged() -> MergedResult:
    return MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            DimensionResult(
                dimension=Dimension.LATENCY,
                status="ok",
                score=1.0,
                confidence=1.0,
                summary="latency 完成",
                metrics={
                    "count": 2,
                    "error_rate": 0.0,
                    "p50_ms": 10,
                    "p95_ms": 20,
                    "p99_ms": 20,
                    "mean_ms": 15.0,
                },
                usage=Usage(samples=2),
            ),
            DimensionResult(
                dimension=Dimension.COST,
                status="ok",
                score=1.0,
                confidence=1.0,
                summary="cost 完成",
                metrics={
                    "input_tokens_total": 3000,
                    "output_tokens_total": 1500,
                    "cost_total_usd": 0.012,
                    "cost_mean_usd": 0.006,
                    "cost_per_1k_requests_usd": 6.0,
                    "estimated_samples": 1,
                },
                usage=Usage(samples=2),
            ),
        ],
        metrics={"overall_score": 1.0},
    )


def probe() -> ProbeSummary:
    return ProbeSummary(
        path="data/runs/run-1/target_probe.jsonl",
        total=2,
        succeeded=2,
        failed=0,
    )


def test_payload_is_version_four_and_includes_probe() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        probe=probe(),
        generated_at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    )

    assert payload["version"] == 4
    assert payload["probe"]["total"] == 2
    assert payload["probe"]["succeeded"] == 2
    assert payload["deferred"] == [
        "证据包上传与 zip/tar 解压、沙箱执行（阶段 4）",
        "L3 Python grader（v1.5）",
        "429/5xx 重试（阶段 5）",
        "流式响应、首 token 延迟与并发探测（v1.5）",
    ]


def test_markdown_renders_probe_and_latency_table() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        probe=probe(),
        generated_at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload, probe_mean_latency_ms=15.0)

    assert "## 目标探测" in markdown
    assert "总请求：2" in markdown
    assert "成功：2，失败：0" in markdown
    assert "平均耗时：15.0 ms" in markdown
    assert "| count | error_rate | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |" in markdown
    assert "| 2 | 0.0 | 10 | 20 | 20 | 15.0 |" in markdown


def test_markdown_renders_cost_table_and_estimated_note() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        probe=probe(),
        generated_at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload)

    assert "| input_tokens | output_tokens | total (¥) | mean (¥) | per 1k (¥) | estimated |" in markdown
    assert "| 3000 | 1500 | ¥0.012000 | ¥0.006000 | ¥6.000000 | 1 |" in markdown
    assert "部分 token 为本地估算（estimated_samples=1）。" in markdown


def test_markdown_notes_missing_probe_for_phase_one_runs() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        generated_at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload)

    assert "（本次运行没有目标探测）" in markdown


def test_markdown_surfaces_failed_dimensions_in_overview() -> None:
    failed = merged().model_copy(update={"metrics": {"overall_score": 1.0, "failed_dimensions": ["cost"]}})
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=failed,
        probe=probe(),
        generated_at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload)

    assert "- 失败维度：成本" in markdown


def test_markdown_omits_failed_dimensions_line_when_none() -> None:
    payload = build_report_payload(
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        probe=probe(),
        generated_at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    )

    markdown = render_markdown(payload)

    assert "失败维度" not in markdown


def test_write_report_reads_probe_records_for_the_mean(tmp_path: Path) -> None:
    paths = run_paths(tmp_path, "run-1")
    store = ProbeStore(paths.target_probe)
    store.reset()
    for index, latency in enumerate([10, 20]):
        store.append(
            ProbeRecord(
                index=index,
                prompt="q",
                started_at=datetime(2026, 10, 28, tzinfo=UTC),
                status_code=200,
                latency_ms=latency,
            )
        )

    report = write_report(
        data_dir=tmp_path,
        run_id="run-1",
        plan=plan(),
        merged=merged(),
        probe=probe(),
        generated_at=datetime(2026, 10, 28, 12, 0, tzinfo=UTC),
    )

    markdown = Path(report.markdown_path).read_text(encoding="utf-8")
    assert "平均耗时：15.0 ms" in markdown
