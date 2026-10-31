import json
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eva_agent.adapters.inputs.target import write_target_workspace
from eva_agent.config.settings import AppConfig, PricingSettings
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import InputKind, PastedInput, RunInput, RunStatus
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.packages import grader_package as gp
from eva_agent.packages.grader_package import build_grader_package, read_grader_package
from eva_agent.packages.inputs import EvidenceSnapshot, PackageInput
from eva_agent.packages.manifest import PackageError
from eva_agent.runtime.artifacts import run_paths
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import budget, fake_config

MEMBERS = (
    "README.md",
    "input.json",
    "plan.json",
    "report.json",
    "snapshots/gradings.json",
)


def _valid_members() -> dict[str, bytes]:
    return {
        "README.md": b"# Grader package\n",
        "input.json": b"{}",
        "plan.json": b"{}",
        "report.json": b'{"version": 4}',
        "snapshots/gradings.json": b"{}",
    }


def _report_path(config: AppConfig, run_row: Any) -> Path:
    return Path(config.storage.data_dir) / "runs" / run_row.id / "report.json"


def _edit_report(config: AppConfig, run_row: Any, edit) -> None:
    path = _report_path(config, run_row)
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


async def test_build_grader_package_writes_every_declared_member(write_run, archive_dir):
    config, run_row = await write_run()
    archive, manifest = await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)
    assert manifest.kind == "grader_package"
    assert manifest.report_version == 4
    assert sorted(manifest.files) == [
        "README.md",
        "input.json",
        "plan.json",
        "report.json",
        "snapshots/gradings.json",
    ]
    package = read_grader_package(archive)
    assert package.plan.dimensions[0].dimension is Dimension.QUALITY
    assert package.inputs.rows[0]["query"] == "q"
    assert package.report["version"] == 4
    assert "usage" in package.report


async def test_build_grader_package_readme_is_chinese(write_run, archive_dir):
    config, run_row = await write_run()
    archive, _ = await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)
    with zipfile.ZipFile(archive) as z:
        readme = z.read("README.md").decode("utf-8")
    assert any("一" <= character <= "鿿" for character in readme)
    for member in ("plan.json", "input.json", "snapshots/gradings.json", "report.json"):
        assert member in readme
    assert "eva replay" in readme
    assert "# Grader package" not in readme


def _priced_config(tmp_path: Path) -> AppConfig:
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "models:\n"
        "  - provider: demo\n"
        "    model: demo-echo\n"
        "    input_per_million: 1.0\n"
        "    output_per_million: 2.0\n",
        encoding="utf-8",
    )
    return fake_config(tmp_path).model_copy(
        update={"pricing": PricingSettings(path=str(pricing))}
    )


async def _completed_target_run(tmp_path: Path, target: TargetSpec):
    """A completed target run whose evidence manifest holds *target* verbatim."""
    config = _priced_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    try:
        factory = create_session_factory(engine)
        runs = RunRepository(factory)
        run_input = RunInput(
            kind=InputKind.TARGET, target_ref=target.name, target_prompts=["hi"]
        )
        plan = EvalPlan(
            objective="evaluate the target",
            evidence_summary="target evidence",
            dimensions=[DimensionPlan(dimension=Dimension.QUALITY, enabled=True)],
            budget=budget(),
        )
        await runs.create("run-1", run_input, budget())
        await runs.set_plan("run-1", plan)
        await runs.set_status("run-1", RunStatus.COMPLETED)
        write_target_workspace("run-1", target, ["hi"], config.storage.data_dir)
        paths = run_paths(config.storage.data_dir, "run-1")
        record = ProbeRecord(
            index=0,
            prompt="hi",
            started_at=datetime.now(UTC),
            status_code=200,
            latency_ms=5,
            response_text="hello",
        )
        paths.target_probe.write_text(record.model_dump_json() + "\n", encoding="utf-8")
        paths.report_json.write_text(
            json.dumps({"version": 4, "dimensions": []}), encoding="utf-8"
        )
        row = await runs.get("run-1")
        assert row is not None
        return config, row
    finally:
        await engine.dispose()


async def test_target_secrets_never_enter_the_package(tmp_path, archive_dir):
    api_key = "sk-live-abcdefgh1234"
    client_secret = "s3cr3t-value"
    target = TargetSpec(
        name="demo",
        kind=TargetKind.HTTP_JSON,
        provider="demo",
        model="demo-echo",
        url=f"https://api.example.com/agent?api_key={api_key}",
        method="POST",
        request_template={"prompt": "{{prompt}}", "client_secret": client_secret},
        response_path="$.text",
    )
    config, run_row = await _completed_target_run(tmp_path, target)

    archive, _ = await build_grader_package(
        config=config, run_row=run_row, out_dir=archive_dir
    )

    with zipfile.ZipFile(archive) as handle:
        members = {name: handle.read(name) for name in handle.namelist()}
    blob = b"".join(members.values())
    assert api_key.encode() not in blob
    assert client_secret.encode() not in blob
    snapshot_target = json.loads(members["input.json"])["evidence"]["target"]
    assert snapshot_target["url"] == "https://api.example.com/agent"
    # The sanitized target must still validate when the package is read back.
    read_grader_package(archive)


async def test_build_grader_package_marks_a_truncated_run_unreplayable(write_run, archive_dir):
    config, run_row = await write_run(truncated_dimensions=["quality"])
    _, manifest = await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)
    assert manifest.replayable is False
    assert any("budget" in blocker for blocker in manifest.replay_blockers)


def _package_input(
    *,
    rows: list[dict[str, str]] | None = None,
    probe_records: list[ProbeRecord] | None = None,
    dimension_limits: dict[str, int],
) -> PackageInput:
    return PackageInput(
        run_input=RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q", response="a")),
        evidence=EvidenceSnapshot(kind=InputKind.PASTE, manifest_sha256="0" * 64),
        rows=list(rows or []),
        probe_records=list(probe_records or []),
        dimension_limits=dimension_limits,
    )


def _probe(index: int) -> ProbeRecord:
    return ProbeRecord(
        index=index,
        prompt=f"p{index}",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        status_code=200,
        latency_ms=10,
        response_text="a",
    )


def test_replay_blockers_reject_an_empty_quality_slice():
    "The quality expert never summarizes zero rows, so export must not claim replayability."
    inputs = _package_input(rows=[], dimension_limits={"quality": 5})
    evidence = inputs.evidence

    blockers = gp._replay_blockers([], {"dimensions": []}, evidence, inputs)

    # ``build_grader_package`` records ``replayable=not blockers``.
    assert blockers == ["quality: the dimension has no inputs to replay"]


def test_replay_blockers_reject_an_empty_probe_slice():
    "Cost and latency raise on an empty probe file; an empty graded-row slice is not their input."
    inputs = _package_input(rows=[], probe_records=[], dimension_limits={"cost": 1})
    evidence = inputs.evidence

    assert gp._replay_blockers([], {"dimensions": []}, evidence, inputs) == [
        "cost: the dimension has no inputs to replay"
    ]

    with_probes = _package_input(rows=[], probe_records=[_probe(0)], dimension_limits={"cost": 1})
    assert gp._replay_blockers([], {"dimensions": []}, evidence, with_probes) == []


def test_read_grader_package_rejects_a_tampered_archive(write_run_archive):
    archive = write_run_archive()
    with pytest.raises(PackageError):
        read_grader_package(archive)


def test_read_grader_package_requires_every_member(write_archive):
    archive = write_archive({"README.md": b"# Grader package\n"})
    with pytest.raises(PackageError, match="missing required members") as info:
        read_grader_package(archive)
    assert "plan.json" in str(info.value)
    assert "snapshots/gradings.json" in str(info.value)


def test_read_grader_package_rejects_a_run_bundle(write_archive):
    archive = write_archive(_valid_members(), kind="run_bundle")
    with pytest.raises(PackageError, match="not a grader package"):
        read_grader_package(archive)


def test_read_grader_package_wraps_an_invalid_member(write_archive):
    archive = write_archive(_valid_members())
    with pytest.raises(PackageError) as info:
        read_grader_package(archive)
    assert "input_value" not in str(info.value)


def test_read_grader_package_rejects_a_missing_declared_member(write_archive):
    members = _valid_members()
    del members["plan.json"]
    archive = write_archive(members, declared=list(MEMBERS))
    with pytest.raises(PackageError, match="missing"):
        read_grader_package(archive)


async def test_build_grader_package_scrubs_absolute_report_paths(write_run, archive_dir):
    config, run_row = await write_run()
    report_path = _report_path(config, run_row)
    absolute_root = str(report_path.parent)

    def edit(payload: dict[str, Any]) -> None:
        payload["probe"] = {
            "path": str(report_path.parent / "target_probe.jsonl"),
            "total": 0,
            "succeeded": 0,
            "failed": 0,
        }
        payload["dimensions"].append(
            {
                "dimension": "security",
                "status": "ok",
                "score": None,
                "confidence": 1.0,
                "summary": "s",
                "metrics": {"active": {"status": "ok", "raw_result_path": str(report_path.parent / "r.json")}},
                "findings": [],
                "evidence": [],
                "usage": {"grader_calls": 0, "samples": 0, "duration_ms": 0},
            }
        )
        payload["findings"].append(
            {
                "id": "f1",
                "severity": "high",
                "category": "c",
                "title": "t",
                "description": "d",
                "evidence": [{"kind": "file", "locator": str(report_path.parent / "r.json")}],
            }
        )

    _edit_report(config, run_row, edit)
    archive, _ = await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)

    needles = {absolute_root, absolute_root.replace("\\", "\\\\"), absolute_root.replace("\\", "/")}
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            text = handle.read(name).decode("utf-8")
            assert not any(needle in text for needle in needles), name

    package = read_grader_package(archive)
    assert package.report["probe"]["path"] == "target_probe.jsonl"
    assert package.report["dimensions"][-1]["metrics"]["active"]["raw_result_path"] == "r.json"
    assert package.report["findings"][-1]["evidence"][0]["locator"] == "r.json"


async def test_build_grader_package_rejects_an_unscrubbed_absolute_path(write_run, archive_dir):
    config, run_row = await write_run()
    report_path = _report_path(config, run_row)
    _edit_report(
        config,
        run_row,
        lambda payload: payload.update({"objective": str(report_path.parent / "secret.txt")}),
    )
    with pytest.raises(PackageError, match="absolute path"):
        await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)


async def test_build_grader_package_replaces_the_archive_only_on_success(
    write_run, archive_dir, monkeypatch
):
    config, run_row = await write_run()
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"{run_row.id}.grader-package.zip"
    destination.write_bytes(b"previous archive")

    def failing_write_zip(root, path, *, manifest):
        # Simulate a failure after the writer has started writing the archive.
        Path(path).write_bytes(b"partial")
        raise OSError(5, "I/O error", str(path))

    monkeypatch.setattr(gp, "write_zip", failing_write_zip)
    with pytest.raises(PackageError, match="cannot write grader package"):
        await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)

    assert destination.read_bytes() == b"previous archive"
    assert [path.name for path in archive_dir.iterdir() if ".tmp-" in path.name] == []


async def test_build_grader_package_wraps_an_unwritable_out_dir(write_run, tmp_path):
    config, run_row = await write_run()
    out_file = tmp_path / "out-is-a-file"
    out_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(PackageError, match="cannot write grader package"):
        await build_grader_package(config=config, run_row=run_row, out_dir=out_file)


async def test_build_grader_package_requires_the_configured_catalog(write_run, archive_dir, tmp_path):
    config, run_row = await write_run()
    config = config.model_copy(
        update={
            "openjudge": config.openjudge.model_copy(
                update={"catalog_path": str(tmp_path / "missing-catalog.yaml")}
            )
        }
    )
    with pytest.raises(PackageError, match="catalog"):
        await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)


async def test_build_grader_package_wraps_an_unreadable_catalog(write_run, archive_dir, tmp_path):
    config, run_row = await write_run()
    catalog_dir = tmp_path / "catalog-dir"
    catalog_dir.mkdir()
    config = config.model_copy(
        update={"openjudge": config.openjudge.model_copy(update={"catalog_path": str(catalog_dir)})}
    )
    with pytest.raises(PackageError, match="catalog"):
        await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)


async def test_build_grader_package_rejects_an_unsafe_run_id(write_run, archive_dir):
    config, run_row = await write_run()
    run_row.id = "../escape"
    with pytest.raises(PackageError, match="invalid run"):
        await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)


async def test_build_and_read_leave_no_temporary_directory(write_run, archive_dir, monkeypatch):
    config, run_row = await write_run()
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(gp.tempfile, "mkdtemp", tracking_mkdtemp)
    archive, _ = await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)
    temp_root = Path(tempfile.gettempdir())
    before = {path for path in temp_root.glob("eva-replay-*")}
    read_grader_package(archive)
    after = {path for path in temp_root.glob("eva-replay-*")}
    assert after == before
    assert created
    assert all(not path.exists() for path in created)
