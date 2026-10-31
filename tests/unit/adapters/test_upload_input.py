import os
import subprocess
from pathlib import Path

import pytest

from eva_agent.adapters.inputs.upload import write_upload_workspace
from eva_agent.domain.run import InputKind


def _make_dir_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"cannot create a directory symlink here: {exc}")


def _make_junction(link: Path, target: Path) -> None:
    if os.name != "nt":  # pragma: no cover - Windows only
        pytest.skip("directory junctions require Windows")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:  # pragma: no cover - platform dependent
        pytest.skip(f"cannot create a directory junction here: {result.stderr or result.stdout}")


def test_upload_workspace_hashes_files(tmp_path: Path) -> None:
    run_id = "run_upload"
    evidence = tmp_path / "runs" / run_id / "evidence"
    (evidence / "dir").mkdir(parents=True)
    (evidence / "a.txt").write_bytes(b"hello")
    (evidence / "dir" / "b.txt").write_bytes(b"world")
    (evidence / "manifest.json").write_text("packaged manifest", encoding="utf-8")

    bundle = write_upload_workspace(run_id, tmp_path)

    assert bundle.kind is InputKind.UPLOAD
    assert [item.path for item in bundle.files] == ["a.txt", "dir/b.txt"]
    assert bundle.files[0].size == 5
    assert bundle.manifest_sha256
    assert (evidence / "manifest.json").read_text(encoding="utf-8") != "packaged manifest"


def test_upload_workspace_requires_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        write_upload_workspace("run_empty", tmp_path)


def test_upload_workspace_rejects_symlink_directory_alias(tmp_path: Path) -> None:
    run_id = "run_symlink_alias"
    evidence = tmp_path / "runs" / run_id / "evidence"
    (evidence / "real").mkdir(parents=True)
    (evidence / "real" / "a.txt").write_bytes(b"hello")
    _make_dir_symlink(evidence / "alias", evidence / "real")

    with pytest.raises(ValueError, match="unsafe path"):
        write_upload_workspace(run_id, tmp_path)


def test_upload_workspace_rejects_junction_directory_alias(tmp_path: Path) -> None:
    run_id = "run_junction_alias"
    evidence = tmp_path / "runs" / run_id / "evidence"
    (evidence / "real").mkdir(parents=True)
    (evidence / "real" / "a.txt").write_bytes(b"hello")
    _make_junction(evidence / "alias", evidence / "real")

    with pytest.raises(ValueError, match="unsafe path"):
        write_upload_workspace(run_id, tmp_path)


def test_upload_workspace_rejects_self_referential_junction(tmp_path: Path) -> None:
    run_id = "run_junction_loop"
    evidence = tmp_path / "runs" / run_id / "evidence"
    (evidence / "real").mkdir(parents=True)
    (evidence / "real" / "a.txt").write_bytes(b"hello")
    _make_junction(evidence / "real" / "loop", evidence / "real")

    with pytest.raises(ValueError, match="unsafe path"):
        write_upload_workspace(run_id, tmp_path)


def test_upload_workspace_rejects_symlink_file(tmp_path: Path) -> None:
    run_id = "run_symlink_file"
    evidence = tmp_path / "runs" / run_id / "evidence"
    evidence.mkdir(parents=True)
    target = evidence / "a.txt"
    target.write_bytes(b"hello")
    try:
        (evidence / "alias.txt").symlink_to(target)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"cannot create a file symlink here: {exc}")

    with pytest.raises(ValueError, match="unsafe path"):
        write_upload_workspace(run_id, tmp_path)


def test_upload_workspace_rejects_symlinked_evidence_root(tmp_path: Path) -> None:
    run_id = "run_symlink_root"
    real = tmp_path / "actual_evidence"
    real.mkdir()
    (real / "a.txt").write_bytes(b"hello")
    evidence = tmp_path / "runs" / run_id / "evidence"
    evidence.parent.mkdir(parents=True)
    _make_dir_symlink(evidence, real)

    with pytest.raises(ValueError, match="unsafe path"):
        write_upload_workspace(run_id, tmp_path)


def test_upload_workspace_rejects_junction_evidence_root(tmp_path: Path) -> None:
    run_id = "run_junction_root"
    real = tmp_path / "actual_evidence"
    real.mkdir()
    (real / "a.txt").write_bytes(b"hello")
    evidence = tmp_path / "runs" / run_id / "evidence"
    evidence.parent.mkdir(parents=True)
    _make_junction(evidence, real)

    with pytest.raises(ValueError, match="unsafe path"):
        write_upload_workspace(run_id, tmp_path)
