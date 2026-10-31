from pathlib import Path

import pytest

from eva_agent.adapters.inputs.upload import write_upload_workspace
from eva_agent.domain.run import InputKind


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
