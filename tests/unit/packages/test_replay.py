import asyncio
import json
from pathlib import Path

import pytest

from eva_agent import cli
from eva_agent.packages.grader_package import build_grader_package, read_grader_package
from eva_agent.packages.manifest import (
    FileEntry,
    PackageError,
    canonical_json,
    extract_zip,
    sha256_file,
    write_manifest,
    write_zip,
)
from eva_agent.packages.replay import replay_package


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
