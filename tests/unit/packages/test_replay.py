import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from eva_agent import cli
from eva_agent.domain.grader import CheckSpec, GraderSpec, ScoreSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.domain.run import InputKind, RunInput
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.graders_dsl.compiler import compile_grader
from eva_agent.graph.experts.cost import build_cost_record, summarize_cost
from eva_agent.graph.experts.latency import build_latency_record, is_successful, summarize_latency
from eva_agent.packages.grader_package import GraderPackage, build_grader_package, read_grader_package
from eva_agent.packages.inputs import EvidenceSnapshot, PackageInput
from eva_agent.packages.manifest import (
    FileEntry,
    PackageError,
    PackageManifest,
    canonical_json,
    extract_zip,
    sha256_file,
    write_manifest,
    write_zip,
)
from eva_agent.packages.replay import replay_package
from eva_agent.packages.snapshots import RecordedOutcome, SnapshotSet
from eva_agent.pricing import ModelPrice, PricingTable
from eva_agent.security.passive import passive_specs, summarize_passive
from tests.support.runtime import budget


def _rebuild_with_score(archive: Path, root: Path, *, score: float) -> Path:
    """Re-zip *archive* with every recorded grader score replaced by *score*.

    The manifest is rebuilt from the rewritten tree, so the archive stays
    structurally valid and the replay sees a consistent package whose snapshot
    no longer matches its report.
    """
    manifest = extract_zip(archive, root)
    snapshot_path = root / "snapshots" / "gradings.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for outcome in payload.get("outcomes") or []:
        outcome["score"] = score
    snapshot_path.write_bytes(canonical_json(payload))
    files = {
        name: FileEntry(sha256=sha256_file(root / name), size=(root / name).stat().st_size)
        for name in manifest.files
    }
    rebuilt = manifest.model_copy(update={"files": files})
    write_manifest(root, rebuilt)
    return write_zip(root, root.with_suffix(".zip"), manifest=rebuilt)


async def test_replay_matches_the_recorded_report(write_run, tmp_path) -> None:
    config, run_row = await write_run()
    archive, manifest = await build_grader_package(config=config, run_row=run_row, out_dir=tmp_path)
    assert manifest.replayable is True
    result = replay_package(read_grader_package(archive))
    assert result.matched is True
    assert result.differences == []
    assert result.compared == ["quality"]


async def test_replay_reports_a_tampered_snapshot(write_run, tmp_path) -> None:
    config, run_row = await write_run()
    archive, _ = await build_grader_package(config=config, run_row=run_row, out_dir=tmp_path)
    tampered = _rebuild_with_score(archive, tmp_path / "tampered", score=0.1)
    result = replay_package(read_grader_package(tampered))
    assert result.matched is False
    assert any(item.path == "dimensions.quality.metrics" for item in result.differences)


async def test_replay_refuses_a_non_replayable_package(write_run, tmp_path) -> None:
    config, run_row = await write_run(truncated_dimensions=["quality"])
    archive, manifest = await build_grader_package(config=config, run_row=run_row, out_dir=tmp_path)
    assert manifest.replayable is False
    with pytest.raises(PackageError, match="not replayable"):
        replay_package(read_grader_package(archive))


def _rebuild_with_report(archive: Path, root: Path, *, report: dict) -> Path:
    """Re-zip *archive* with ``report.json`` replaced by *report*.

    The manifest is rebuilt from the rewritten tree, so the archive is hash-valid
    and the only defect is the report's shape.
    """
    manifest = extract_zip(archive, root)
    (root / "report.json").write_bytes(canonical_json(report))
    files = {
        name: FileEntry(sha256=sha256_file(root / name), size=(root / name).stat().st_size)
        for name in manifest.files
    }
    rebuilt = manifest.model_copy(update={"files": files})
    write_manifest(root, rebuilt)
    return write_zip(root, root.with_suffix(".zip"), manifest=rebuilt)


MALFORMED_REPORTS = [
    pytest.param({"version": 4, "dimensions": [1]}, id="dimension-entry-not-an-object"),
    pytest.param({"version": 4, "dimensions": {"quality": {}}}, id="dimensions-not-a-list"),
    pytest.param(
        {"version": 4, "dimensions": [{"dimension": "quality", "metrics": [1]}]},
        id="metrics-not-an-object",
    ),
    pytest.param(
        {"version": 4, "dimensions": [{"dimension": "quality", "findings": 1}]},
        id="findings-not-a-list",
    ),
    pytest.param(
        {"version": 4, "dimensions": [{"dimension": "quality", "error": [1]}]},
        id="error-not-a-string",
    ),
    pytest.param(
        {"version": 4, "dimensions": [{"metrics": {}}]},
        id="missing-dimension-name",
    ),
    pytest.param(
        {"version": 4, "dimensions": [{"dimension": "security", "metrics": {"active": 1}}]},
        id="active-not-an-object",
    ),
]


@pytest.mark.parametrize("report", MALFORMED_REPORTS)
async def test_read_grader_package_rejects_a_malformed_report(write_run, tmp_path, report) -> None:
    config, run_row = await write_run()
    archive, _ = await build_grader_package(config=config, run_row=run_row, out_dir=tmp_path)
    malformed = _rebuild_with_report(archive, tmp_path / "malformed", report=report)
    with pytest.raises(PackageError, match="report.json"):
        read_grader_package(malformed)


async def test_main_reports_a_malformed_report_without_a_traceback(
    write_run, tmp_path, capsys
) -> None:
    config, run_row = await write_run()
    archive, _ = await build_grader_package(config=config, run_row=run_row, out_dir=tmp_path)
    malformed = _rebuild_with_report(
        archive, tmp_path / "malformed", report={"version": 4, "dimensions": [1]}
    )
    out_dir = tmp_path / "out"
    # ``main`` owns its event loop (``asyncio.run``), so run it off this test's loop.
    code = await asyncio.to_thread(
        cli.main, ["replay", str(malformed), "--out", str(out_dir)]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err
    assert not (out_dir / "replay.json").exists()


TARGET = TargetSpec(
    name="demo-target",
    kind=TargetKind.OPENAI_COMPATIBLE,
    provider="demo-provider",
    model="demo-model",
    base_url="http://localhost:1",
)
ROWS = [
    {"query": "q0", "response": "a0", "context": "", "reference": ""},
    {"query": "q1", "response": "a1", "context": "", "reference": ""},
]


def _evidence() -> EvidenceSnapshot:
    return EvidenceSnapshot(kind=InputKind.TARGET, target=TARGET, manifest_sha256="0" * 64)


def _plan(*dimensions: DimensionPlan) -> EvalPlan:
    return EvalPlan(
        objective="replay a hand-built package",
        evidence_summary="fixture",
        dimensions=list(dimensions),
        budget=budget(),
    )


def _package(
    plan: EvalPlan,
    *,
    rows: list[dict[str, str]] | None = None,
    outcomes: list[RecordedOutcome] | None = None,
    probe_records: list[ProbeRecord] | None = None,
    prices: list[ModelPrice] | None = None,
    dimensions: list[dict[str, Any]] | None = None,
) -> GraderPackage:
    """A replayable package built from models only: no database, cache, or config file."""
    package_rows = ROWS if rows is None else rows
    inputs = PackageInput(
        run_input=RunInput(kind=InputKind.TARGET, target_ref=TARGET.name, target_prompts=["prompt"]),
        evidence=_evidence(),
        rows=package_rows,
        source_indexes=[None] * len(package_rows),
        probe_records=list(probe_records or []),
        prices=list(prices or []),
    )
    manifest = PackageManifest(
        kind="grader_package",
        run_id="run-1",
        created_at="2026-01-01T00:00:00Z",
        files={},
    )
    return GraderPackage(
        manifest=manifest,
        plan=plan,
        inputs=inputs,
        snapshots=SnapshotSet(outcomes=list(outcomes or [])),
        report={"dimensions": list(dimensions or [])},
    )


def _threshold_spec(field: str, *, value: float) -> GraderSpec:
    return GraderSpec(
        level="declarative",
        name=f"{field}-check",
        input_map={field: f"$.{field}"},
        checks=[CheckSpec(name="check", type="numeric_threshold", field=field, op="<=", value=value)],
        score=ScoreSpec(aggregation="weighted_mean", weights={"check": 1.0}),
    )


def _declarative_output(spec: GraderSpec, records: list[dict[str, Any]]) -> GraderRunOutput:
    """The output the live declarative grader produces for *records*."""
    grader = compile_grader(spec)
    outcomes: list[GraderSampleOutcome] = []
    for index, record in enumerate(records):
        score, checks = grader.evaluate(record)
        failed = [check.reason for check in checks if not check.passed]
        outcomes.append(
            GraderSampleOutcome(
                grader=spec.name,
                sample_index=index,
                score=score,
                raw_score=score,
                reason="; ".join(failed),
            )
        )
    return GraderRunOutput(outcomes=outcomes)


def _passive_entry(rows: list[dict[str, str]], outcomes: list[RecordedOutcome]) -> dict[str, Any]:
    """The report entry the security expert writes for *outcomes*."""
    specs = passive_specs(rows)
    expected = summarize_passive(
        GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(
                    grader=item.spec_name,
                    sample_index=item.sample_index,
                    score=item.score,
                    raw_score=item.raw_score,
                    reason=item.reason or "",
                    error=item.error,
                )
                for item in outcomes
            ]
        ),
        specs=specs,
        rows=rows,
        evidence=_evidence().as_bundle(),
    )
    return {
        "dimension": "security",
        "score": expected.score,
        "confidence": expected.confidence,
        "metrics": {"passive": expected.metrics, "active": {"status": "not_applicable"}},
    }


def _passive_outcomes(*scores: float) -> list[RecordedOutcome]:
    return [
        RecordedOutcome(
            dimension=Dimension.SECURITY,
            spec_name="harmfulness",
            sample_index=index,
            score=score,
            raw_score=score,
            reason="",
        )
        for index, score in enumerate(scores)
    ]


def test_replay_matches_a_security_package_with_passive_outcomes() -> None:
    entry = _passive_entry(ROWS, _passive_outcomes(0.9, 0.7))
    plan = _plan(DimensionPlan(dimension=Dimension.SECURITY, enabled=True))

    result = replay_package(_package(plan, outcomes=_passive_outcomes(0.9, 0.7), dimensions=[entry]))

    assert result.matched is True
    assert result.differences == []
    assert result.compared == ["security"]


def test_replay_reports_a_tampered_security_snapshot() -> None:
    entry = _passive_entry(ROWS, _passive_outcomes(0.9, 0.7))
    plan = _plan(DimensionPlan(dimension=Dimension.SECURITY, enabled=True))

    result = replay_package(_package(plan, outcomes=_passive_outcomes(0.1, 0.7), dimensions=[entry]))

    assert result.matched is False
    assert any(item.path == "dimensions.security.metrics.passive" for item in result.differences)
    assert all(item.path.startswith("dimensions.security.") for item in result.differences)


def test_replay_matches_a_cost_package_using_packaged_prices(monkeypatch) -> None:
    def _no_config_pricing(*_args: Any, **_kwargs: Any) -> PricingTable:
        raise AssertionError("replay must use the packaged price row, not config/pricing.yaml")

    monkeypatch.setattr(PricingTable, "load", staticmethod(_no_config_pricing))
    price = ModelPrice(
        provider=TARGET.provider,
        model=TARGET.model,
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    )
    records = [
        ProbeRecord(
            index=0,
            prompt="p0",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status_code=200,
            latency_ms=10,
            response_text="a0",
            input_tokens=1000,
            output_tokens=500,
            usage_source="target",
        ),
        ProbeRecord(
            index=1,
            prompt="p1",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status_code=500,
            latency_ms=20,
            response_text="",
            input_tokens=2000,
            output_tokens=100,
            usage_source="target",
            error="upstream error",
        ),
    ]
    costs = [
        price.cost_usd(input_tokens=record.input_tokens or 0, output_tokens=record.output_tokens or 0)
        for record in records
    ]
    priced_rows = [build_cost_record(record, cost) for record, cost in zip(records, costs, strict=True)]
    spec = _threshold_spec("cost_usd", value=1.0)
    expected = summarize_cost(
        records=records,
        rows=priced_rows,
        costs=costs,
        outcomes=[_declarative_output(spec, priced_rows)],
        specs=[spec],
        sample_size=len(records),
    )
    plan = _plan(DimensionPlan(dimension=Dimension.COST, enabled=True, graders=[spec]))

    result = replay_package(
        _package(
            plan,
            probe_records=records,
            prices=[price],
            dimensions=[expected.model_dump(mode="json")],
        )
    )

    assert result.matched is True
    assert result.differences == []
    assert result.compared == ["cost"]
    assert expected.metrics["sample_size"] == len(records)
    assert expected.metrics["priced_samples"] == len(records)


def test_replay_matches_a_latency_package_with_successful_rows_only() -> None:
    records = [
        ProbeRecord(
            index=0,
            prompt="p0",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status_code=200,
            latency_ms=100,
            response_text="a0",
        ),
        ProbeRecord(
            index=1,
            prompt="p1",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status_code=500,
            latency_ms=250,
            response_text="",
            error="upstream error",
        ),
        ProbeRecord(
            index=2,
            prompt="p2",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            status_code=200,
            latency_ms=900,
            response_text="a2",
        ),
    ]
    spec = _threshold_spec("latency_ms", value=1000.0)
    selected = records[:2]
    successful_rows = [build_latency_record(record) for record in selected if is_successful(record)]
    expected = summarize_latency(
        records=selected,
        outcomes=[_declarative_output(spec, successful_rows)],
        specs=[spec],
        sample_size=2,
    )
    plan = _plan(DimensionPlan(dimension=Dimension.LATENCY, enabled=True, sample_size=2, graders=[spec]))

    result = replay_package(
        _package(plan, probe_records=records, dimensions=[expected.model_dump(mode="json")])
    )

    assert result.matched is True
    assert result.differences == []
    assert result.compared == ["latency"]
    assert expected.metrics["count"] == 2
    assert expected.metrics["success_count"] == 1
    assert expected.metrics["failure_count"] == 1
    assert expected.metrics["sample_size"] == 2


def test_replay_refuses_a_quality_dimension_with_no_rows() -> None:
    spec = GraderSpec(level="builtin", name="relevance", builtin_id="relevance")
    plan = _plan(DimensionPlan(dimension=Dimension.QUALITY, enabled=True, graders=[spec]))

    package = _package(plan, rows=[], dimensions=[{"dimension": "quality"}])

    with pytest.raises(PackageError, match="quality"):
        replay_package(package)


def test_replay_refuses_a_cost_dimension_with_no_probe_records() -> None:
    price = ModelPrice(
        provider=TARGET.provider,
        model=TARGET.model,
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    )
    spec = _threshold_spec("cost_usd", value=1.0)
    plan = _plan(DimensionPlan(dimension=Dimension.COST, enabled=True, graders=[spec]))

    package = _package(plan, probe_records=[], prices=[price], dimensions=[{"dimension": "cost"}])

    with pytest.raises(PackageError, match="cost"):
        replay_package(package)
