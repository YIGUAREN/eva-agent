import json
from pathlib import Path

from eva_agent.domain.plan import Dimension
from eva_agent.domain.run import DatasetRef, InputKind, RunInput, RunStatus
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.manager import RunManager
from eva_agent.server.routes.reports import public_report_usage
from eva_agent.server.schemas import sanitize_public_payload
from tests.support.runtime import budget, dependencies, fake_config


async def test_dataset_run_meets_phase_one_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "qa.csv"
    source.write_text(
        "query,response,reference\nq1,a1,r1\nq2,a2,r2\n",
        encoding="utf-8",
    )
    run_input = RunInput(
        kind=InputKind.DATASET,
        objective="评估回答质量",
        dataset=DatasetRef(path=str(source), format="csv"),
    )

    async with RunManager.open(fake_config(tmp_path), dependencies=dependencies(tmp_path)) as manager:
        run_id = await manager.start(run_input, budget(), run_id="run-accept")
        waiting = await manager.status(run_id)
        paths = run_paths(tmp_path / "data", run_id)

        assert waiting.status is RunStatus.WAITING_APPROVAL
        assert waiting.plan is not None
        quality = next(item for item in waiting.plan.dimensions if item.dimension is Dimension.QUALITY)
        assert any(spec.level == "llm_rubric" for spec in quality.graders)
        assert paths.plan_draft.is_file()
        assert (paths.evidence / "manifest.json").is_file()
        assert (paths.evidence / "dataset.csv").is_file()

        completed = await manager.approve(run_id, approved=True)
        report = await manager.report(run_id)

    assert completed.status is RunStatus.COMPLETED
    assert paths.plan_approved.is_file()
    assert paths.report_markdown.is_file()
    assert paths.report_json.is_file()
    assert report["status"] == "completed"
    assert report["dimensions"][0]["dimension"] == "quality"
    assert report["dimensions"][0]["score"] == 0.8
    assert {item["dimension"] for item in report["skipped"]} == {"latency", "cost", "security"}
    assert report["findings"] == []
    # The CLI (and any unenforced manager) keeps the phase 1-3 contract: no
    # reservation, no gate, and a report that says the budget was not enforced.
    usage = report["usage"]
    assert usage["budget_enforced"] is False
    assert usage["estimated"] is False
    # Provenance still counts grading calls on an unenforced run (the dataset
    # is 2 samples x 2 graders), so only its shape is part of the contract.
    assert isinstance(usage["cached"], int) and usage["cached"] >= 0
    assert isinstance(usage["fresh"], int) and usage["fresh"] >= 0
    assert report["truncated_dimensions"] == []
    assert report["finalize_overrun"] is False
    # Spend and paths are admin-only: the public projection keeps only the four
    # non-spend usage keys and replaces absolute paths wholesale.
    public_usage = public_report_usage(report)
    assert set(public_usage) == {"estimated", "cached", "fresh", "budget_enforced"}
    # Mirror the public report route: sanitize paths, then whitelist usage.
    public = {**sanitize_public_payload(report), "usage": public_usage}
    public_text = json.dumps(public, ensure_ascii=False)
    assert "spent_usd" not in public_text
    assert str(tmp_path) not in public_text
