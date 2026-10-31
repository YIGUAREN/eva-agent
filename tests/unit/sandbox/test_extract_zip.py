import hashlib
import stat
import zipfile
from pathlib import Path

import pytest

from eva_agent.sandbox.extract import ExtractionError, ExtractionLimits, detect_format, extract_archive


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


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("a\x00b.txt", "invalid_path"),
        ("a\\b.txt", "invalid_path"),
    ],
)
def test_rejects_names_zipfile_would_sanitize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, reason: str
) -> None:
    """NUL and backslash names must fail closed even when they reach us.

    CPython's zipfile truncates at NUL and, on Windows, rewrites backslashes
    to slashes while reading, so its public reader never exposes these names.
    Disable that read-time normalization to simulate an archive produced by a
    foreign tool and prove the extractor still rejects them.
    """
    monkeypatch.setattr(zipfile, "_sanitize_filename", lambda value: value)
    package = _zip(tmp_path / "p.bin", {name: b"x"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == reason
    assert not (tmp_path / "out").exists()


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
