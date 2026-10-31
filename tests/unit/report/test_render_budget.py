import json
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.result import DimensionResult, MergedResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.report.render import REPORT_VERSION, build_report_payload, render_markdown


def _plan() -> EvalPlan:
    return EvalPlan(
        objective="o",
        evidence_summary="e",
        dimensions=[DimensionPlan(dimension=Dimension.QUALITY, enabled=True, sample_size=1, graders=[])],
        skipped=[
            SkippedDimension(dimension=Dimension.LATENCY, reason="off"),
            SkippedDimension(dimension=Dimension.COST, reason="off"),
            SkippedDimension(dimension=Dimension.SECURITY, reason="off"),
        ],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
    )


def _merged() -> MergedResult:
    return MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            DimensionResult(
                dimension=Dimension.QUALITY,
                status="partial",
                score=0.5,
                confidence=0.5,
                summary="partial",
                usage=Usage(grader_calls=2, samples=2, cached=1, fresh=1),
                truncated_reason="budget",
            )
        ],
        skipped=_plan().skipped,
        metrics={"overall_score": 0.5},
    )


def _payload(
    usage: dict[str, object] | None,
    finalize_overrun: bool = True,
    *,
    merged: MergedResult | None = None,
) -> dict:
    return build_report_payload(
        run_id="run-1",
        plan=_plan(),
        merged=merged or _merged(),
        generated_at=datetime(2026, 10, 31, tzinfo=UTC),
        usage=usage,
        finalize_overrun=finalize_overrun,
    )


def _merged_with_quality_metrics(metrics: dict[str, object]) -> MergedResult:
    base = _merged()
    return base.model_copy(
        update={"results": [base.results[0].model_copy(update={"metrics": metrics})]}
    )


def test_report_v4_carries_truncation_usage_and_spend() -> None:
    payload = _payload(
        {
            "estimated": True,
            "cached": 1,
            "fresh": 1,
            "budget_enforced": True,
            "spent_usd": 0.9,
        }
    )
    assert REPORT_VERSION == 4
    assert payload["version"] == 4
    assert payload["truncated_dimensions"] == ["quality"]
    assert payload["usage"] == {
        "estimated": True,
        "cached": 1,
        "fresh": 1,
        "budget_enforced": True,
        "spent_usd": 0.9,
    }
    assert payload["finalize_overrun"] is True
    assert payload["truncated_sample_counts"] == [
        {
            "dimension": "quality",
            "sample_count": None,
            "completed_samples": None,
            "missing_samples": None,
        }
    ]


def test_report_without_a_usage_argument_still_emits_zeroed_usage() -> None:
    payload = _payload(None, finalize_overrun=False)
    assert payload["usage"] == {
        "estimated": False,
        "cached": 0,
        "fresh": 0,
        "budget_enforced": False,
        "spent_usd": 0.0,
    }
    assert payload["finalize_overrun"] is False


def test_malformed_usage_fields_are_coerced() -> None:
    payload = _payload(
        {
            "estimated": None,
            "cached": None,
            "fresh": None,
            "budget_enforced": None,
            "spent_usd": None,
        }
    )
    assert payload["usage"] == {
        "estimated": False,
        "cached": 0,
        "fresh": 0,
        "budget_enforced": False,
        "spent_usd": 0.0,
    }


@pytest.mark.parametrize("value", ["abc", -1, True])
def test_malformed_cached_and_fresh_counts_coerce_to_zero(value: object) -> None:
    payload = _payload(
        {
            "estimated": False,
            "cached": value,
            "fresh": value,
            "budget_enforced": False,
            "spent_usd": 0,
        }
    )
    assert payload["usage"]["cached"] == 0
    assert payload["usage"]["fresh"] == 0


def test_non_boolean_usage_flags_coerce_to_false() -> None:
    payload = _payload(
        {
            "estimated": "false",
            "cached": 1,
            "fresh": 1,
            "budget_enforced": "false",
            "spent_usd": 0,
        }
    )
    assert payload["usage"]["estimated"] is False
    assert payload["usage"]["budget_enforced"] is False


def test_decimal_spend_stays_json_serializable() -> None:
    payload = _payload(
        {
            "estimated": False,
            "cached": 1,
            "fresh": 1,
            "budget_enforced": True,
            "spent_usd": Decimal("1.250000"),
        }
    )
    assert payload["usage"]["spent_usd"] == 1.25
    json.dumps(payload)


def test_truncated_sample_counts_derive_completion_and_gaps() -> None:
    payload = _payload(
        {"estimated": False, "cached": 0, "fresh": 0, "budget_enforced": True},
        merged=_merged_with_quality_metrics({"sample_count": 4, "scored_sample_count": 1}),
    )
    assert payload["truncated_sample_counts"] == [
        {
            "dimension": "quality",
            "sample_count": 4,
            "completed_samples": 1,
            "missing_samples": 3,
        }
    ]


def test_truncated_sample_counts_read_security_passive_counts() -> None:
    merged = MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            DimensionResult(
                dimension=Dimension.SECURITY,
                status="partial",
                score=0.5,
                confidence=0.5,
                summary="sec",
                usage=Usage(),
                truncated_reason="budget",
                metrics={"passive": {"sample_count": 4, "scored_rows": 2}},
            )
        ],
        skipped=[],
        metrics={},
    )
    payload = _payload(
        {"estimated": False, "cached": 0, "fresh": 0, "budget_enforced": True},
        merged=merged,
    )
    assert payload["truncated_sample_counts"] == [
        {
            "dimension": "security",
            "sample_count": 4,
            "completed_samples": 2,
            "missing_samples": 2,
        }
    ]


def test_truncated_sample_counts_are_null_for_unrecorded_security_active() -> None:
    merged = MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            DimensionResult(
                dimension=Dimension.SECURITY,
                status="partial",
                score=None,
                confidence=0.0,
                summary="sec",
                usage=Usage(),
                truncated_reason="budget",
                metrics={
                    "passive": {"status": "not_applicable", "sample_count": 0},
                    "active": {"status": "failed"},
                },
            )
        ],
        skipped=[],
        metrics={},
    )
    payload = _payload(
        {"estimated": False, "cached": 0, "fresh": 0, "budget_enforced": True},
        merged=merged,
    )
    assert payload["truncated_sample_counts"] == [
        {
            "dimension": "security",
            "sample_count": None,
            "completed_samples": None,
            "missing_samples": None,
        }
    ]


def test_truncated_latency_and_cost_counts_stay_na() -> None:
    merged = MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            DimensionResult(
                dimension=Dimension.LATENCY,
                status="partial",
                score=1.0,
                confidence=1.0,
                summary="latency",
                usage=Usage(),
                truncated_reason="budget",
                metrics={"count": 2, "success_count": 2},
            ),
            DimensionResult(
                dimension=Dimension.COST,
                status="partial",
                score=1.0,
                confidence=1.0,
                summary="cost",
                usage=Usage(),
                truncated_reason="budget",
                metrics={"count": 2, "priced_samples": 2},
            ),
        ],
        skipped=[],
        metrics={},
    )
    payload = _payload(
        {"estimated": False, "cached": 0, "fresh": 0, "budget_enforced": True},
        merged=merged,
    )
    assert payload["truncated_sample_counts"] == [
        {
            "dimension": "cost",
            "sample_count": None,
            "completed_samples": None,
            "missing_samples": None,
        },
        {
            "dimension": "latency",
            "sample_count": None,
            "completed_samples": None,
            "missing_samples": None,
        },
    ]


def test_markdown_states_the_source_split_with_the_admin_spend_line() -> None:
    payload = _payload(
        {
            "estimated": False,
            "cached": 1,
            "fresh": 1,
            "budget_enforced": True,
            "spent_usd": 0.9,
        }
    )
    markdown = render_markdown(payload)
    assert "- 预算执行：已强制" in markdown
    assert "- 截断维度：质量" in markdown
    assert "- 收尾超支：是" in markdown
    assert "- 评分来源：缓存 1，新评分 1" in markdown
    assert "- 已花费：¥0.900000" in markdown


def test_markdown_renders_the_truncation_sample_table() -> None:
    payload = _payload(
        {"estimated": False, "cached": 0, "fresh": 0, "budget_enforced": True},
        merged=_merged_with_quality_metrics({"sample_count": 4, "scored_sample_count": 1}),
    )
    markdown = render_markdown(payload)
    assert "## 截断样本" in markdown
    assert "| 维度 | 计划/总样本 | 已完成样本 | 缺失样本 |" in markdown
    assert "| 质量 | 4 | 1 | 3 |" in markdown


def test_markdown_renders_na_for_unrecorded_sample_counts() -> None:
    payload = _payload(
        {"estimated": False, "cached": 0, "fresh": 0, "budget_enforced": True}
    )
    markdown = render_markdown(payload)
    assert "| 质量 | n/a | n/a | n/a |" in markdown


def test_markdown_renders_the_active_scan_raw_result_path_for_admins() -> None:
    merged = _merged().model_copy(
        update={
            "results": [
                DimensionResult(
                    dimension=Dimension.SECURITY,
                    status="ok",
                    score=0.5,
                    confidence=1.0,
                    summary="sec",
                    metrics={
                        "active": {
                            "kind": "skill_scan",
                            "status": "completed",
                            "raw_result_path": "C:\\data\\runs\\run-1\\aig\\abc.json",
                            "dataset": [],
                            "num_prompts": 0,
                            "asr": None,
                            "jailbreak": 0,
                            "total": 0,
                            "severity_counts": {},
                        }
                    },
                    usage=Usage(),
                )
            ]
        }
    )
    payload = build_report_payload(
        run_id="run-1",
        plan=_plan(),
        merged=merged,
        generated_at=datetime(2026, 10, 31, tzinfo=UTC),
    )
    markdown = render_markdown(payload)
    assert "扫描种类：`skill_scan`" in markdown
    assert "原始结果：`C:\\data\\runs\\run-1\\aig\\abc.json`" in markdown
    assert json.dumps(payload)  # the payload stays JSON-serializable
