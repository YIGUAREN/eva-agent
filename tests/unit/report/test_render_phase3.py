from datetime import UTC, datetime

from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.result import DimensionResult, MergedResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.report.render import build_report_payload, render_markdown


def payload() -> dict:
    plan = EvalPlan(
        objective="阶段 3",
        evidence_summary="target",
        dimensions=[
            DimensionPlan(dimension=Dimension.QUALITY, enabled=True, sample_size=2),
            DimensionPlan(dimension=Dimension.SECURITY, enabled=True, sample_size=2),
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.LATENCY, reason="x"),
            SkippedDimension(dimension=Dimension.COST, reason="x"),
        ],
        budget=Budget(max_cost_usd=1, max_wall_clock_s=60, max_target_requests=10),
    )
    merged = MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            DimensionResult(
                dimension=Dimension.QUALITY,
                status="ok",
                score=0.8,
                confidence=1.0,
                summary="quality ok",
                metrics={
                    "sample_count": 2,
                    "scored_sample_count": 2,
                    "by_grader": {"relevance": 0.8},
                    "errors": [],
                },
                usage=Usage(samples=2),
            ),
            DimensionResult(
                dimension=Dimension.SECURITY,
                status="partial",
                score=0.5,
                confidence=1.0,
                summary="security partial",
                metrics={
                    "passive": {
                        "by_grader": {"harmfulness": 0.8, "code_security": 0.6},
                        "scored_outcomes": 4,
                        "expected_outcomes": 4,
                        "scored_rows": 2,
                        "total_rows": 2,
                        "errors": [],
                    },
                    "active": {
                        "status": "completed",
                        "dataset": ["d"],
                        "num_prompts": 2,
                        "asr": 0.5,
                        "jailbreak": 1,
                        "total": 2,
                        "severity_counts": {"high": 1},
                        "vulnerabilities": [
                            {
                                "vulnerability": "Harmful Content",
                                "asr": 0.5,
                                "severity": "high",
                                "jailbreak": 1,
                                "total": 2,
                            }
                        ],
                        "degraded": False,
                        "refused_reason": None,
                    },
                    "score_cap": 0.5,
                },
                usage=Usage(samples=2),
            ),
        ],
        metrics={"overall_score": 0.65},
    )
    return build_report_payload(
        run_id="run-1",
        plan=plan,
        merged=merged,
        generated_at=datetime(2026, 10, 30, tzinfo=UTC),
    )


def test_report_version_is_three_and_deferrals_drop_phase_three() -> None:
    data = payload()
    assert data["version"] == 4
    assert all("阶段 3" not in item for item in data["deferred"])


def test_markdown_renders_quality_and_security_tables() -> None:
    markdown = render_markdown(payload())
    assert "**质量评分**" in markdown
    assert "| relevance | 0.800 | 2 | 0 |" in markdown
    assert "**被动检查**" in markdown
    assert "| harmfulness | 0.800 |" in markdown
    assert "**主动扫描**" in markdown
    assert "| completed | d | 2 | 0.500 | 1 / 2 | high=1 |" in markdown
    assert "| Harmful Content | 0.500 | high | 1 / 2 |" in markdown


def test_markdown_shows_refusal_reason() -> None:
    data = payload()
    security = data["dimensions"][1]
    security["metrics"]["active"].update(
        {
            "status": "refused",
            "refused_reason": "not_admin",
            "asr": None,
            "jailbreak": 0,
            "total": 0,
            "vulnerabilities": [],
            "severity_counts": {},
        }
    )
    markdown = render_markdown(data)
    assert "拒绝原因：`not_admin`" in markdown


def test_markdown_does_not_fabricate_security_metrics_when_empty() -> None:
    data = payload()
    security = data["dimensions"][1]
    security["metrics"] = {}
    security["status"] = "failed"
    security["score"] = None
    security["error"] = "no usable security score"
    markdown = render_markdown(data)
    assert "| None | - | 0 | n/a | 0 / 0 | - |" not in markdown
    assert "安全检查未完成：没有可用指标。" in markdown
    assert "- 错误：no usable security score" in markdown
