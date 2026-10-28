from pathlib import Path

from eva_agent.domain.plan import Dimension
from eva_agent.domain.run import DatasetRef, InputKind, RunInput, RunStatus
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.manager import RunManager
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
