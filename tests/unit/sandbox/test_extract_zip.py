import hashlib
import stat
import zipfile
import zlib
from pathlib import Path

import pytest

from eva_agent.sandbox.extract import (
    ExtractionError,
    ExtractionLimits,
    _reject_unsafe_unicode_path_extra,
    detect_format,
    extract_archive,
)

_UNICODE_PATH_EXTRA_ID = 0x7075


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_detect_zip_magic(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {"a.txt": b"hello"})
    assert detect_format(package) == "zip"


def test_extract_zip_writes_manifest(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {"b.txt": b"two", "dir/a.txt": b"one"})
    destination = tmp_path / "out"

    manifest = extract_archive(package, destination, limits=ExtractionLimits())

    assert manifest == {
        "format": "zip",
        "files": [
            {"path": "b.txt", "size": 3, "sha256": hashlib.sha256(b"two").hexdigest()},
            {"path": "dir/a.txt", "size": 3, "sha256": hashlib.sha256(b"one").hexdigest()},
        ],
        "total_bytes": 6,
    }
    assert (destination / "dir" / "a.txt").read_bytes() == b"one"


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("../escape.txt", "path_traversal"),
        ("/absolute.txt", "absolute_path"),
    ],
)
def test_rejects_unsafe_names(tmp_path: Path, name: str, reason: str) -> None:
    package = _zip(tmp_path / "p.bin", {name: b"x"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == reason
    assert not (tmp_path / "out").exists()


def _patch_member_name(path: Path, stored: bytes, declared: bytes) -> None:
    """Rewrite a member name in both the local header and central directory.

    ``zipfile`` sanitizes names on read (NUL truncation; backslash rewriting on
    Windows), so the raw declared name has to be injected into the archive
    bytes after creation. The substitution keeps byte lengths equal so every
    other offset stays valid.
    """
    assert len(stored) == len(declared)
    data = path.read_bytes()
    assert data.count(stored) == 2
    path.write_bytes(data.replace(stored, declared))


@pytest.mark.parametrize(
    ("stored", "declared"),
    [
        (b"aXb.txt", b"a\x00b.txt"),
        (b"a-b.txt", b"a\\b.txt"),
    ],
)
def test_rejects_names_zipfile_would_sanitize(tmp_path: Path, stored: bytes, declared: bytes) -> None:
    """Raw declared NUL and backslash names must fail closed."""
    package = _zip(tmp_path / "p.bin", {stored.decode("ascii"): b"x"})
    _patch_member_name(package, stored, declared)
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


def _unicode_path_extra(raw_name: bytes, unicode_name: bytes) -> bytes:
    """Build a 0x7075 Unicode-path extra field with a valid name CRC."""
    payload = b"\x01" + zlib.crc32(raw_name).to_bytes(4, "little") + unicode_name
    return _UNICODE_PATH_EXTRA_ID.to_bytes(2, "little") + len(payload).to_bytes(2, "little") + payload


def _zip_with_unicode_path(path: Path, unicode_name: bytes) -> Path:
    """Create a zip whose 0x7075 field is patched into both headers in place.

    The writer emits the extra field in the local header and the central
    directory, so patching the equally sized placeholder payload rewrites both
    copies without shifting any offsets.
    """
    raw_name = b"safe.txt"
    placeholder = b"xxxxxxx"
    assert len(placeholder) == len(unicode_name)
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo(raw_name.decode("ascii"))
        info.extra = _unicode_path_extra(raw_name, placeholder)
        archive.writestr(info, b"x")
    data = path.read_bytes()
    assert data.count(placeholder) == 2
    path.write_bytes(data.replace(placeholder, unicode_name))
    return path


@pytest.mark.parametrize(
    "declared",
    [b"a\\b.txt", b"a\x00b.txt"],
    ids=["backslash", "nul"],
)
def test_rejects_unsafe_unicode_path_extra(tmp_path: Path, declared: bytes) -> None:
    """The raw 0x7075 payload must be rejected even though zipfile sanitizes it."""
    package = _zip_with_unicode_path(tmp_path / "p.bin", declared)
    # Sanity check: zipfile itself does apply the extra field to ``filename``.
    with zipfile.ZipFile(package) as archive:
        info = archive.infolist()[0]
        assert info.orig_filename == "safe.txt"
        assert info.filename != "safe.txt"
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "extra",
    [
        # 0x7075 whose data is shorter than version + name_crc32.
        _UNICODE_PATH_EXTRA_ID.to_bytes(2, "little") + (2).to_bytes(2, "little") + b"\x01\x00",
        # 0x7075 with a valid header but a non-UTF-8 payload.
        _UNICODE_PATH_EXTRA_ID.to_bytes(2, "little") + (6).to_bytes(2, "little") + b"\x01\x00\x00\x00\x00\xff",
        # Declared field size overruns the extra buffer; the parser must not over-read.
        _UNICODE_PATH_EXTRA_ID.to_bytes(2, "little") + (100).to_bytes(2, "little") + b"\x01\x00\x00\x00\x00",
    ],
    ids=["short-header", "invalid-utf8", "overlong-size"],
)
def test_rejects_malformed_unicode_path_extra(extra: bytes) -> None:
    info = zipfile.ZipInfo("safe.txt")
    info.extra = extra
    with pytest.raises(ExtractionError) as caught:
        _reject_unsafe_unicode_path_extra(info)
    assert caught.value.reason == "invalid_path"


def test_ignores_trailing_extra_bytes_too_short_for_a_header() -> None:
    info = zipfile.ZipInfo("safe.txt")
    info.extra = b"\x75\x70"
    _reject_unsafe_unicode_path_extra(info)


def test_rejects_symlink_and_special_modes(tmp_path: Path) -> None:
    for mode, reason in [(stat.S_IFLNK, "symlink"), (stat.S_IFCHR, "special_file")]:
        package = tmp_path / f"{reason}.zip"
        with zipfile.ZipFile(package, "w") as archive:
            info = zipfile.ZipInfo("link")
            info.external_attr = (mode | 0o777) << 16
            archive.writestr(info, b"x")
        with pytest.raises(ExtractionError) as caught:
            extract_archive(package, tmp_path / "out")
        assert caught.value.reason == reason
        assert not (tmp_path / "out").exists()


def _mark_encrypted(path: Path) -> None:
    """Set the encrypted flag in both zip headers without needing a password writer."""
    data = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        index = 0
        while (index := data.find(signature, index)) != -1:
            data[index + flag_offset] |= 0x1
            index += 4
    path.write_bytes(bytes(data))


def test_rejects_encrypted_entry(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {"a.txt": b"x"})
    _mark_encrypted(package)
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "encrypted"


def test_rejects_per_file_limit(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {"big.bin": b"x" * 11})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out", limits=ExtractionLimits(max_file_bytes=10))
    assert caught.value.reason == "file_too_large"


def test_rejects_declared_total(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {"a": b"x" * 8, "b": b"y" * 8})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out", limits=ExtractionLimits(max_total_bytes=10))
    assert caught.value.reason == "total_too_large"


def test_rejects_file_count_and_depth(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {"a": b"1", "b": b"2"})
    with pytest.raises(ExtractionError):
        extract_archive(package, tmp_path / "out", limits=ExtractionLimits(max_files=1))

    deep = _zip(tmp_path / "deep.bin", {"a/b/c.txt": b"1"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(deep, tmp_path / "out2", limits=ExtractionLimits(max_depth=2))
    assert caught.value.reason == "too_deep"


def test_rejects_unsupported_magic(tmp_path: Path) -> None:
    package = tmp_path / "p.bin"
    package.write_bytes(b"not an archive")
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "unsupported_format"


def test_failed_extraction_leaves_no_output(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {"ok.txt": b"ok", "../bad": b"bad"})
    destination = tmp_path / "out"
    with pytest.raises(ExtractionError):
        extract_archive(package, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "name",
    ["NUL", "CON", "nul.txt", "COM1", "LPT0", "LPT9.log", "CONIN$", "aux.tar.gz"],
)
def test_rejects_windows_reserved_device_names(tmp_path: Path, name: str) -> None:
    package = _zip(tmp_path / "p.bin", {name: b"x"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "name",
    [
        "dir/a:b.txt",
        "dir/a<b.txt",
        "dir/a>b.txt",
        'dir/a"b.txt',
        "dir/a|b.txt",
        "dir/a?b.txt",
        "dir/a*b.txt",
        "dir/bad\x01.txt",
        "dir/bad\x7f.txt",
        "dir/trailing.",
        "dir/trailing ",
    ],
)
def test_rejects_windows_invalid_characters_and_trailing(tmp_path: Path, name: str) -> None:
    package = _zip(tmp_path / "p.bin", {name: b"x"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


def test_rejects_entry_declared_on_later_volume(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {"a.txt": b"x"})
    data = bytearray(package.read_bytes())
    index = data.find(b"PK\x01\x02")
    assert index != -1
    data[index + 34 : index + 36] = (1).to_bytes(2, "little")
    package.write_bytes(bytes(data))
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "unsupported_format"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("offset", [4, 6])
def test_rejects_eocd_disk_numbers(tmp_path: Path, offset: int) -> None:
    package = _zip(tmp_path / "p.bin", {"a.txt": b"x"})
    data = bytearray(package.read_bytes())
    index = data.rfind(b"PK\x05\x06")
    assert index != -1
    data[index + offset : index + offset + 2] = (1).to_bytes(2, "little")
    package.write_bytes(bytes(data))
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "unsupported_format"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("disk", [0, 1])
def test_rejects_appended_data_after_eocd(tmp_path: Path, disk: int) -> None:
    """Trailing bytes must not let the EOCD validation fall through."""
    package = _zip(tmp_path / "p.bin", {"a.txt": b"x"})
    data = bytearray(package.read_bytes())
    index = data.rfind(b"PK\x05\x06")
    assert index != -1
    data[index + 4 : index + 6] = disk.to_bytes(2, "little")
    package.write_bytes(bytes(data) + b"J")
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "unsupported_format"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("comment", [b"", b"evidence package comment"])
def test_accepts_zip_with_or_without_comment(tmp_path: Path, comment: bytes) -> None:
    package = tmp_path / "p.bin"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("a.txt", b"x")
        archive.comment = comment
    manifest = extract_archive(package, tmp_path / "out")
    assert [item["path"] for item in manifest["files"]] == ["a.txt"]
    assert (tmp_path / "out" / "a.txt").read_bytes() == b"x"


def test_accepts_stored_member_containing_zip_signature(tmp_path: Path) -> None:
    """Stored member data may itself contain an EOCD signature (nested zip)."""
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("inner.txt", b"inside")
    nested_bytes = inner.read_bytes()

    outer = tmp_path / "outer.bin"
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("nested/inner.zip", nested_bytes)
    # The outer archive legitimately holds two EOCD signatures: the nested
    # archive's and its own.
    assert outer.read_bytes().count(b"PK\x05\x06") == 2
    with zipfile.ZipFile(outer) as archive:
        assert archive.namelist() == ["nested/inner.zip"]

    manifest = extract_archive(outer, tmp_path / "out")

    assert [item["path"] for item in manifest["files"]] == ["nested/inner.zip"]
    assert (tmp_path / "out" / "nested" / "inner.zip").read_bytes() == nested_bytes


def test_still_rejects_appended_data_with_nested_signature(tmp_path: Path) -> None:
    """Trailing bytes after the last EOCD must stay rejected despite earlier signatures."""
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("inner.txt", b"inside")
    outer = tmp_path / "outer.bin"
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("nested/inner.zip", inner.read_bytes())
    package = tmp_path / "p.bin"
    package.write_bytes(outer.read_bytes() + b"J")

    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "unsupported_format"
    assert not (tmp_path / "out").exists()


def test_maps_filesystem_errors_to_invalid_path(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {"a": b"x", "a/b": b"y"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


@pytest.mark.filterwarnings("ignore:Duplicate name")
def test_rejects_duplicate_member_paths(tmp_path: Path) -> None:
    package = tmp_path / "p.bin"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("a.txt", b"good")
        archive.writestr("a.txt", b"evil")
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


def test_rejects_case_folded_duplicate_member_paths(tmp_path: Path) -> None:
    package = tmp_path / "p.bin"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("A.txt", b"good")
        archive.writestr("a.txt", b"evil")
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


def test_skips_root_directory_member(tmp_path: Path) -> None:
    package = tmp_path / "p.bin"
    with zipfile.ZipFile(package, "w") as archive:
        root = zipfile.ZipInfo("./")
        root.external_attr = (stat.S_IFDIR | 0o755) << 16
        archive.writestr(root, b"")
        archive.writestr("a.txt", b"x")
    manifest = extract_archive(package, tmp_path / "out")
    assert [item["path"] for item in manifest["files"]] == ["a.txt"]
    assert (tmp_path / "out" / "a.txt").read_bytes() == b"x"


def test_rejects_file_member_with_empty_normalized_name(tmp_path: Path) -> None:
    package = _zip(tmp_path / "p.bin", {".": b"x"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


def test_replace_failure_removes_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _zip(tmp_path / "p.bin", {"a.txt": b"x"})
    destination = tmp_path / "out"

    def _fail(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("eva_agent.sandbox.extract.os.replace", _fail)
    with pytest.raises(OSError):
        extract_archive(package, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".out.tmp-*"))
