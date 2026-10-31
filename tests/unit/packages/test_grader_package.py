import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

from eva_agent.config.settings import AppConfig
from eva_agent.domain.plan import Dimension
from eva_agent.packages import grader_package as gp
from eva_agent.packages.grader_package import build_grader_package, read_grader_package
from eva_agent.packages.manifest import PackageError

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


async def test_build_grader_package_marks_a_truncated_run_unreplayable(write_run, archive_dir):
    config, run_row = await write_run(truncated_dimensions=["quality"])
    _, manifest = await build_grader_package(config=config, run_row=run_row, out_dir=archive_dir)
    assert manifest.replayable is False
    assert any("budget" in blocker for blocker in manifest.replay_blockers)


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
