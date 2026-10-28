from datetime import UTC, datetime

import pytest

from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.result import Approval
from eva_agent.runtime.artifacts import (
    read_plan_draft,
    run_paths,
    write_plan_approved,
    write_plan_draft,
)


def plan() -> EvalPlan:
    return EvalPlan(
        objective="评估回答质量",
        evidence_summary="1 条样本",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.QUALITY,
                enabled=True,
                graders=[GraderSpec(level="builtin", name="relevance", builtin_id="relevance")],
            )
        ],
        skipped=[
            SkippedDimension(dimension=Dimension.LATENCY, reason="未实现"),
            SkippedDimension(dimension=Dimension.COST, reason="未实现"),
            SkippedDimension(dimension=Dimension.SECURITY, reason="未实现"),
        ],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1),
    )


def test_run_paths_layout(tmp_path) -> None:
    paths = run_paths(tmp_path, "run-1")

    assert paths.evidence == tmp_path / "runs" / "run-1" / "evidence"
    assert paths.plan_draft.name == "plan.draft.yaml"
    assert paths.report_json.name == "report.json"


def test_plan_draft_round_trips(tmp_path) -> None:
    paths = run_paths(tmp_path, "run-1")

    write_plan_draft(paths, plan())

    assert paths.plan_draft.is_file()
    assert read_plan_draft(paths) == plan()


def test_plan_draft_does_not_overwrite_by_default(tmp_path) -> None:
    paths = run_paths(tmp_path, "run-1")
    write_plan_draft(paths, plan())
    original = paths.plan_draft.read_text(encoding="utf-8")

    write_plan_draft(paths, plan().model_copy(update={"objective": "另一个目标"}))

    assert paths.plan_draft.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("run_id", ["..", "../../evil", "/abs", "a/b", "C:\\x", ""])
def test_run_paths_rejects_unsafe_run_id(tmp_path, run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid run_id"):
        run_paths(tmp_path, run_id)


def test_plan_draft_overwrites_when_asked(tmp_path) -> None:
    paths = run_paths(tmp_path, "run-1")
    write_plan_draft(paths, plan())

    write_plan_draft(paths, plan().model_copy(update={"objective": "另一个目标"}), overwrite=True)

    assert read_plan_draft(paths).objective == "另一个目标"


def test_plan_approved_writes_decision_time(tmp_path) -> None:
    paths = run_paths(tmp_path, "run-1")
    approval = Approval(plan=plan(), decided_at=datetime(2026, 10, 27, 12, 0, tzinfo=UTC))

    write_plan_approved(paths, approval)

    text = paths.plan_approved.read_text(encoding="utf-8")
    assert "2026-10-27T12:00:00" in text
    assert "评估回答质量" in text
