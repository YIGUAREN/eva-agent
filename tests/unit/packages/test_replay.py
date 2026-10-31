import json
from pathlib import Path

import pytest

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
