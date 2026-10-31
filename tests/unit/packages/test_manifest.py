import errno
import json
import re
import zipfile
from pathlib import Path
from typing import IO, Any

import pytest

from eva_agent.packages.manifest import (
    MANIFEST_NAME,
    MAX_PACKAGE_FILES,
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


class _ExpandingReader:
    """A zip member reader that yields bytes beyond the archive's declared size."""

    def __init__(self, inner: IO[bytes], extra: bytes) -> None:
        self._inner = inner
        self._extra = extra

    def read(self, size: int = -1) -> bytes:
        data = self._inner.read(size)
        if not data and self._extra:
            data, self._extra = self._extra, b""
        return data

    def __enter__(self) -> "_ExpandingReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self._inner.close()


def _manifest(files: dict[str, FileEntry]) -> PackageManifest:
    return PackageManifest(
        kind="grader_package",
        run_id="run-1",
        created_at="2026-10-31T00:00:00+00:00",
        files=files,
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


def _assert_path_free(message: str) -> None:
    assert re.search(r"[A-Za-z]:\\", message) is None, message
    assert "eva-import-" not in message, message
    assert "eva-bundle-" not in message, message
    assert "Traceback" not in message, message


def test_missing_manifest_error_does_not_leak_the_host_path(tmp_path) -> None:
    with pytest.raises(PackageError) as caught:
        read_manifest(tmp_path)
    _assert_path_free(str(caught.value))


def test_missing_archive_error_does_not_leak_the_host_path(tmp_path) -> None:
    source = tmp_path / "nope.zip"
    with pytest.raises(PackageError) as caught:
        extract_zip(source, tmp_path / "out")
    _assert_path_free(str(caught.value))


def test_write_zip_error_does_not_leak_the_destination_path(tmp_path, monkeypatch) -> None:
    source = tmp_path / "package"
    manifest = _tree(source)
    destination = tmp_path / "out" / "package.zip"

    class _FailingZip:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise PermissionError(errno.EACCES, "Permission denied", str(destination))

    monkeypatch.setattr(zipfile, "ZipFile", _FailingZip)
    with pytest.raises(PackageError) as caught:
        write_zip(source, destination, manifest=manifest)
    message = str(caught.value)
    assert "Permission denied" in message
    assert str(destination) not in message
    _assert_path_free(message)


def test_invalid_manifest_error_does_not_echo_hostile_values(tmp_path) -> None:
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"kind": "C:\\secret\\evil.json"}), encoding="utf-8"
    )
    with pytest.raises(PackageError) as caught:
        read_manifest(tmp_path)
    _assert_path_free(str(caught.value))


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


def test_extract_zip_counts_directory_entries_toward_the_file_cap(tmp_path) -> None:
    archive = tmp_path / "directories.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for index in range(MAX_PACKAGE_FILES):
            handle.writestr(f"dir-{index}/", "")
        handle.writestr("payload.json", "{}")
    with pytest.raises(PackageError, match=f"{MAX_PACKAGE_FILES + 1} files"):
        extract_zip(archive, tmp_path / "out")


def test_extract_zip_rejects_case_insensitive_duplicate_members(tmp_path) -> None:
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("plan.json", "{}")
        handle.writestr("PLAN.JSON", "{}")
    with pytest.raises(PackageError, match="duplicate package member"):
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


def test_extract_zip_leaves_no_partial_tree_on_failure(tmp_path) -> None:
    source = tmp_path / "package"
    manifest = _tree(source)
    archive = write_zip(source, tmp_path / "package.zip", manifest=manifest)
    (source / "plan.json").write_text('{"plan": 9}', encoding="utf-8")
    write_zip(source, archive, manifest=manifest)
    destination = tmp_path / "out"
    with pytest.raises(PackageError, match="hash mismatch for plan.json"):
        extract_zip(archive, destination)
    assert not destination.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["package", "package.zip"]
    (source / "plan.json").write_text('{"plan": 1}', encoding="utf-8")
    write_zip(source, archive, manifest=manifest)
    assert extract_zip(archive, destination) == manifest


def test_extract_zip_leaves_a_preexisting_destination_when_the_source_is_not_a_zip(tmp_path) -> None:
    destination = tmp_path / "out"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    source = tmp_path / "not-a-zip.zip"
    source.write_bytes(b"this is not a zip archive")
    with pytest.raises(PackageError, match="not a zip package"):
        extract_zip(source, destination)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["not-a-zip.zip", "out"]


def _stored_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("plan.json", "{}")


def test_extract_zip_translates_an_encrypted_member(tmp_path) -> None:
    archive = tmp_path / "encrypted.zip"
    _stored_zip(archive)
    raw = bytearray(archive.read_bytes())
    local = raw.index(b"PK\x03\x04")
    central = raw.index(b"PK\x01\x02")
    raw[local + 6] |= 0x1
    raw[central + 8] |= 0x1
    archive.write_bytes(raw)
    with pytest.raises(PackageError, match="not a zip package"):
        extract_zip(archive, tmp_path / "out")


def test_extract_zip_translates_unsupported_compression(tmp_path) -> None:
    archive = tmp_path / "method.zip"
    _stored_zip(archive)
    raw = bytearray(archive.read_bytes())
    local = raw.index(b"PK\x03\x04")
    central = raw.index(b"PK\x01\x02")
    raw[local + 8 : local + 10] = (99).to_bytes(2, "little")
    raw[central + 10 : central + 12] = (99).to_bytes(2, "little")
    archive.write_bytes(raw)
    with pytest.raises(PackageError, match="not a zip package"):
        extract_zip(archive, tmp_path / "out")


def test_extract_zip_translates_a_bad_crc(tmp_path) -> None:
    archive = tmp_path / "crc.zip"
    _stored_zip(archive)
    raw = bytearray(archive.read_bytes())
    local = raw.index(b"PK\x03\x04")
    data_start = local + 30 + len("plan.json")
    raw[data_start] ^= 0xFF
    archive.write_bytes(raw)
    with pytest.raises(PackageError, match="not a zip package"):
        extract_zip(archive, tmp_path / "out")


def test_extract_zip_translates_an_invalid_utf8_member_name(tmp_path) -> None:
    archive = tmp_path / "bad-name.zip"
    _stored_zip(archive)
    raw = bytearray(archive.read_bytes())
    central = raw.index(b"PK\x01\x02")
    name_length = int.from_bytes(raw[central + 28 : central + 30], "little")
    name_start = central + 46
    raw[central + 9] |= 0x08  # general-purpose flag bit 0x800: filename is UTF-8
    raw[name_start : name_start + name_length] = b"\xff" * name_length
    archive.write_bytes(raw)
    with pytest.raises(PackageError, match="not a zip package"):
        extract_zip(archive, tmp_path / "out")


@pytest.mark.parametrize("member", ["C:escape.json", "NUL", "sub/NUL"])
def test_extract_zip_rejects_unsafe_windows_member_names(tmp_path, member: str) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    archive = workdir / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, "{}")
    destination = workdir / "out"
    with pytest.raises(PackageError, match="invalid member name"):
        extract_zip(archive, destination)
    assert not destination.exists()
    assert sorted(path.name for path in workdir.iterdir()) == ["evil.zip"]


def test_extract_zip_caps_bytes_actually_written(tmp_path, monkeypatch) -> None:
    source = tmp_path / "package"
    manifest = _tree(source)
    archive = write_zip(source, tmp_path / "package.zip", manifest=manifest)
    monkeypatch.setattr("eva_agent.packages.manifest.MAX_PACKAGE_BYTES", 4096)
    original_open = zipfile.ZipFile.open

    def expanding_open(self: zipfile.ZipFile, name: Any, *args: Any, **kwargs: Any) -> Any:
        handle = original_open(self, name, *args, **kwargs)
        if getattr(name, "filename", name) == "plan.json":
            return _ExpandingReader(handle, b"x" * 5000)
        return handle

    monkeypatch.setattr(zipfile.ZipFile, "open", expanding_open)
    with pytest.raises(PackageError, match="expands past 4096 bytes"):
        extract_zip(archive, tmp_path / "out")


def test_verify_tree_rejects_a_manifest_key_that_is_a_directory(tmp_path) -> None:
    (tmp_path / "staging").mkdir()
    manifest = _manifest({"staging": FileEntry(sha256="0" * 64, size=0)})
    with pytest.raises(PackageError, match="member is not a file: staging"):
        verify_tree(tmp_path, manifest)


@pytest.mark.parametrize("name", ["../escape.json", "C:escape.json"])
def test_verify_tree_rejects_a_manifest_key_outside_the_root(tmp_path, name: str) -> None:
    manifest = _manifest({name: FileEntry(sha256="0" * 64, size=0)})
    with pytest.raises(PackageError, match="invalid member name"):
        verify_tree(tmp_path, manifest)


def test_write_zip_enforces_the_member_cap(tmp_path) -> None:
    files = {f"file-{index}.json": FileEntry(sha256="0" * 64, size=0) for index in range(MAX_PACKAGE_FILES)}
    manifest = _manifest(files)
    with pytest.raises(PackageError, match=f"{MAX_PACKAGE_FILES + 1} files"):
        write_zip(tmp_path / "source", tmp_path / "many.zip", manifest=manifest)


def test_write_zip_enforces_the_size_cap(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = b"x" * 5000
    (source / "big.bin").write_bytes(payload)
    manifest = _manifest({"big.bin": FileEntry(sha256=sha256_bytes(payload), size=len(payload))})
    write_manifest(source, manifest)
    monkeypatch.setattr("eva_agent.packages.manifest.MAX_PACKAGE_BYTES", 4096)
    with pytest.raises(PackageError, match="expands past 4096 bytes"):
        write_zip(source, tmp_path / "big.zip", manifest=manifest)


@pytest.mark.parametrize("name", [MANIFEST_NAME, "Manifest.json", "MANIFEST.JSON"])
def test_write_zip_rejects_listing_the_manifest_itself(tmp_path, name: str) -> None:
    manifest = _manifest({name: FileEntry(sha256="0" * 64, size=0)})
    with pytest.raises(PackageError, match="may not list"):
        write_zip(tmp_path / "source", tmp_path / "out.zip", manifest=manifest)


def test_write_zip_rejects_listing_a_directory(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "sub").mkdir()
    manifest = _manifest({"sub": FileEntry(sha256="0" * 64, size=0)})
    write_manifest(source, manifest)
    with pytest.raises(PackageError, match="not a file"):
        write_zip(source, tmp_path / "out.zip", manifest=manifest)


def test_write_zip_translates_a_write_failure(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    manifest = _tree(source)

    def failing_write(self: zipfile.ZipFile, filename: Any, *args: Any, **kwargs: Any) -> None:
        # A real OS failure carries ``errno``/``strerror``; ``_reason`` reports only
        # that text so no filesystem path can reach the message.
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(zipfile.ZipFile, "write", failing_write)
    with pytest.raises(PackageError, match="cannot write package member: disk full"):
        write_zip(source, tmp_path / "out.zip", manifest=manifest)


@pytest.mark.parametrize("name", ["a//b.json", "a/./b.json", "./plan.json", "dir/"])
def test_manifest_keys_must_be_normalized(tmp_path, name: str) -> None:
    manifest = _manifest({name: FileEntry(sha256="0" * 64, size=0)})
    with pytest.raises(PackageError, match="invalid member name"):
        write_manifest(tmp_path / "root", manifest)
    with pytest.raises(PackageError, match="invalid member name"):
        write_zip(tmp_path / "source", tmp_path / "out.zip", manifest=manifest)


@pytest.mark.parametrize("name", ["../escape.json", "C:escape.json"])
def test_write_zip_rejects_unsafe_member_names(tmp_path, name: str) -> None:
    manifest = _manifest({name: FileEntry(sha256="0" * 64, size=0)})
    with pytest.raises(PackageError, match="invalid member name"):
        write_zip(tmp_path / "source", tmp_path / "out.zip", manifest=manifest)


@pytest.mark.parametrize("name", ["../escape.json", "C:escape.json", "C:/secret"])
def test_write_manifest_rejects_unsafe_keys(tmp_path, name: str) -> None:
    manifest = _manifest({name: FileEntry(sha256="0" * 64, size=0)})
    with pytest.raises(PackageError, match="invalid member name"):
        write_manifest(tmp_path / "root", manifest)


@pytest.mark.parametrize("name", ["../escape.json", "C:escape.json"])
def test_read_manifest_rejects_unsafe_keys(tmp_path, name: str) -> None:
    manifest = _manifest({name: FileEntry(sha256="0" * 64, size=0)})
    root = tmp_path / "root"
    root.mkdir()
    (root / MANIFEST_NAME).write_bytes(canonical_json(manifest.model_dump(mode="json")))
    with pytest.raises(PackageError, match="invalid member name"):
        read_manifest(root)


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
