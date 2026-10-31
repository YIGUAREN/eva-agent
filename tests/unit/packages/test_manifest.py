import json
import zipfile
from pathlib import Path

import pytest

from eva_agent.packages.manifest import (
    MANIFEST_NAME,
    FileEntry,
    PackageError,
    PackageManifest,
    canonical_json,
    extract_zip,
    read_manifest,
    sha256_bytes,
    verify_tree,
    write_manifest,
    write_zip,
)


def _tree(root: Path) -> PackageManifest:
    (root / "snapshots").mkdir(parents=True)
    (root / "plan.json").write_text('{"plan": 1}', encoding="utf-8")
    (root / "snapshots" / "gradings.json").write_text('{"outcomes": []}', encoding="utf-8")
    files = {
        name: FileEntry(
            sha256=sha256_bytes((root / name).read_bytes()),
            size=len((root / name).read_bytes()),
        )
        for name in ("plan.json", "snapshots/gradings.json")
    }
    manifest = PackageManifest(
        kind="grader_package",
        run_id="run-1",
        created_at="2026-10-31T00:00:00+00:00",
        report_version=4,
        files=files,
    )
    write_manifest(root, manifest)
    return manifest


def test_manifest_round_trips_through_canonical_json(tmp_path) -> None:
    manifest = _tree(tmp_path)
    raw = (tmp_path / MANIFEST_NAME).read_bytes()
    assert raw == canonical_json(manifest.model_dump(mode="json"))
    assert read_manifest(tmp_path) == manifest


def test_verify_tree_rejects_a_tampered_member(tmp_path) -> None:
    _tree(tmp_path)
    (tmp_path / "plan.json").write_text('{"plan": 2}', encoding="utf-8")
    with pytest.raises(PackageError, match="hash mismatch for plan.json"):
        verify_tree(tmp_path, read_manifest(tmp_path))


def test_verify_tree_rejects_a_missing_member(tmp_path) -> None:
    _tree(tmp_path)
    (tmp_path / "plan.json").unlink()
    with pytest.raises(PackageError, match="missing"):
        verify_tree(tmp_path, read_manifest(tmp_path))


def test_verify_tree_rejects_an_extra_member(tmp_path) -> None:
    _tree(tmp_path)
    (tmp_path / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PackageError, match="unexpected files"):
        verify_tree(tmp_path, read_manifest(tmp_path))


def test_read_manifest_rejects_a_missing_manifest(tmp_path) -> None:
    with pytest.raises(PackageError, match="cannot read"):
        read_manifest(tmp_path)


def test_zip_round_trip_verifies(tmp_path) -> None:
    source = tmp_path / "package"
    manifest = _tree(source)
    archive = write_zip(source, tmp_path / "out" / "package.zip", manifest=manifest)
    target = tmp_path / "extracted"
    assert extract_zip(archive, target) == manifest
    assert (target / "snapshots" / "gradings.json").is_file()


def test_extract_zip_rejects_path_traversal(tmp_path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.json", "{}")
    with pytest.raises(PackageError, match="invalid member name"):
        extract_zip(archive, tmp_path / "out")


def test_extract_zip_rejects_a_symlink_member(tmp_path) -> None:
    archive = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link.json")
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "/etc/passwd")
    with pytest.raises(PackageError, match="symlink"):
        extract_zip(archive, tmp_path / "out")


def test_extract_zip_rejects_a_tampered_member(tmp_path) -> None:
    source = tmp_path / "package"
    manifest = _tree(source)
    archive = write_zip(source, tmp_path / "package.zip", manifest=manifest)
    (source / "plan.json").write_text('{"plan": 9}', encoding="utf-8")
    write_zip(source, archive, manifest=manifest)
    with pytest.raises(PackageError, match="hash mismatch for plan.json"):
        extract_zip(archive, tmp_path / "out")


def test_manifest_is_frozen_and_forbids_extra_fields(tmp_path) -> None:
    with pytest.raises(ValueError):
        PackageManifest(
            kind="grader_package",
            run_id="run-1",
            created_at="x",
            files={},
            extra=1,  # pyright: ignore[reportCallIssue]
        )
    payload = json.loads(canonical_json({"b": 1, "a": [2, 3]}))
    assert list(payload) == ["a", "b"]
