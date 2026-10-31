import json
from datetime import UTC, datetime

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


def _payload(usage: dict[str, object], finalize_overrun: bool = True) -> dict:
    return build_report_payload(
        run_id="run-1",
        plan=_plan(),
        merged=_merged(),
        generated_at=datetime(2026, 10, 31, tzinfo=UTC),
        usage=usage,
        finalize_overrun=finalize_overrun,
    )


def test_report_v4_carries_truncation_and_usage() -> None:
    payload = _payload(
        {"estimated": True, "cached": 1, "fresh": 1, "budget_enforced": True}
    )
    assert REPORT_VERSION == 4
    assert payload["version"] == 4
    assert payload["truncated_dimensions"] == ["quality"]
    assert payload["usage"] == {
        "estimated": True,
        "cached": 1,
        "fresh": 1,
        "budget_enforced": True,
    }
    assert payload["finalize_overrun"] is True


def test_report_without_a_usage_argument_still_emits_zeroed_usage() -> None:
    payload = _payload(None, finalize_overrun=False)  # type: ignore[arg-type]
    assert payload["usage"] == {
        "estimated": False,
        "cached": 0,
        "fresh": 0,
        "budget_enforced": False,
    }
    assert payload["finalize_overrun"] is False


def test_markdown_states_the_source_split_without_money() -> None:
    payload = _payload(
        {"estimated": False, "cached": 1, "fresh": 1, "budget_enforced": True}
    )
    markdown = render_markdown(payload)
    assert "- 预算执行：已强制" in markdown
    assert "- 截断维度：quality" in markdown
    assert "- 收尾超支：是" in markdown
    assert "- 评分来源：缓存 1，新评分 1" in markdown
    assert "spent" not in markdown


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
