import gzip
import io
import tarfile
from pathlib import Path

import pytest

from eva_agent.sandbox.extract import (
    ExtractionError,
    ExtractionLimits,
    _LimitedReader,
    detect_format,
    extract_archive,
)


def _tar(path: Path, members: dict[str, bytes], *, compression: str = "") -> Path:
    mode = "w:gz" if compression == "gz" else "w"
    with tarfile.open(path, mode) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _tar_directories(path: Path, names: list[str]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name in names:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
    return path


def _tar_header(name: str, size: int, typeflag: bytes) -> bytes:
    """Build one 512-byte ustar header with a correct octal size and checksum."""
    encoded = name.encode("ascii")
    assert len(encoded) <= 100
    header = bytearray(512)
    header[0 : len(encoded)] = encoded
    header[100:108] = b"0000644\x00"
    header[108:116] = b"0000000\x00"
    header[116:124] = b"0000000\x00"
    header[124:136] = f"{size:011o}\x00".encode("ascii")
    header[136:148] = b"00000000000\x00"
    header[148:156] = b" " * 8
    header[156:157] = typeflag
    header[257:263] = b"ustar\x00"
    header[263:265] = b"00"
    checksum = sum(header)
    header[148:156] = f"{checksum:06o}\x00 ".encode("ascii")
    return bytes(header)


@pytest.mark.parametrize("compression", ["", "gz"])
def test_extract_tar_writes_manifest(tmp_path: Path, compression: str) -> None:
    package = _tar(tmp_path / "p.bin", {"a.txt": b"one", "dir/b.txt": b"two"}, compression=compression)
    destination = tmp_path / "out"

    manifest = extract_archive(package, destination, limits=ExtractionLimits())

    assert manifest["format"] == "tar"
    assert [item["path"] for item in manifest["files"]] == ["a.txt", "dir/b.txt"]
    assert manifest["total_bytes"] == 6
    assert (destination / "dir" / "b.txt").read_bytes() == b"two"


def test_detect_tar_variants(tmp_path: Path) -> None:
    assert detect_format(_tar(tmp_path / "plain", {"a": b"x"})) == "tar"
    assert detect_format(_tar(tmp_path / "gz", {"a": b"x"}, compression="gz")) == "tar"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "char"])
def test_rejects_special_tar_members(tmp_path: Path, kind: str) -> None:
    package = tmp_path / "p.tar"
    with tarfile.open(package, "w") as archive:
        info = tarfile.TarInfo("member")
        if kind == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
        elif kind == "hardlink":
            info.type = tarfile.LNKTYPE
            info.linkname = "target"
        elif kind == "fifo":
            info.type = tarfile.FIFOTYPE
        else:
            info.type = tarfile.CHRTYPE
        archive.addfile(info)
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == {"symlink": "symlink", "hardlink": "hardlink"}.get(kind, "special_file")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["../x", "/abs/x", "a\\b"])
def test_rejects_unsafe_tar_names(tmp_path: Path, name: str) -> None:
    package = _tar(tmp_path / "p.tar", {name: b"x"})
    with pytest.raises(ExtractionError):
        extract_archive(package, tmp_path / "out")


def test_rejects_tar_declared_size(tmp_path: Path) -> None:
    package = _tar(tmp_path / "p.tar", {"a": b"x" * 11})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out", limits=ExtractionLimits(max_file_bytes=10))
    assert caught.value.reason == "file_too_large"


def test_rejects_corrupt_tar(tmp_path: Path) -> None:
    package = tmp_path / "p.bin"
    package.write_bytes(b"\x1f\x8b" + b"not-gzip")
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "corrupt"
    assert not (tmp_path / "out").exists()


def test_limited_reader_bounds_cumulative_decompression() -> None:
    reader = _LimitedReader(io.BytesIO(b"x" * 100), 10, total_limit=20)

    assert reader.read(10) == b"x" * 10
    assert reader.read(10) == b"x" * 10
    with pytest.raises(ExtractionError) as caught:
        reader.read(10)

    assert caught.value.reason == "total_too_large"


def test_limited_reader_caps_unbounded_reads() -> None:
    reader = _LimitedReader(io.BytesIO(b"x" * 100), 10)

    assert reader.read() == b"x" * 10


@pytest.mark.parametrize(
    "payload",
    [
        b"\x1f\x8b",
        gzip.compress(b"x" * 1024)[:15],
    ],
)
def test_rejects_truncated_gzip_as_corrupt(tmp_path: Path, payload: bytes) -> None:
    """A truncated gzip stream must be a rejection, never a server error."""
    package = tmp_path / "p.bin"
    package.write_bytes(payload)

    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")

    assert caught.value.reason == "corrupt"
    assert not (tmp_path / "out").exists()


def test_rejects_tar_depth(tmp_path: Path) -> None:
    package = _tar(tmp_path / "p.tar", {"a/b/c": b"x"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out", limits=ExtractionLimits(max_depth=2))
    assert caught.value.reason == "too_deep"


def test_rejects_tar_file_count(tmp_path: Path) -> None:
    package = _tar(tmp_path / "p.tar", {"a": b"1", "b": b"2"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out", limits=ExtractionLimits(max_files=1))
    assert caught.value.reason == "too_many_files"


def test_rejects_directory_only_tar_with_too_many_members(tmp_path: Path) -> None:
    """Directory entries count toward max_files even though nothing is written."""
    package = _tar_directories(tmp_path / "p.tar", [f"d{index}" for index in range(5)])
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out", limits=ExtractionLimits(max_files=2))
    assert caught.value.reason == "too_many_files"
    assert not (tmp_path / "out").exists()


def test_rejects_directory_only_tar_without_regular_files(tmp_path: Path) -> None:
    package = _tar_directories(tmp_path / "p.tar", ["only"])
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "rejected"
    assert not (tmp_path / "out").exists()


def test_rejects_pax_header_with_an_absurd_declared_size(tmp_path: Path) -> None:
    """A tiny archive declaring a huge pax block must fail closed, not exhaust memory."""
    package = tmp_path / "p.tar"
    # typeflag "x" is a pax extended header; declare 2 GiB but ship no payload.
    package.write_bytes(_tar_header("pax", 2**31, b"x") + b"\x00" * 1024)

    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")

    assert caught.value.reason in {"file_too_large", "corrupt"}
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["NUL", "nul.txt", "COM1", "LPT0", "CONIN$"])
def test_rejects_windows_reserved_device_names(tmp_path: Path, name: str) -> None:
    package = _tar(tmp_path / "p.tar", {name: b"x"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["dir/a:b.txt", "dir/bad\x01.txt", "dir/trailing."])
def test_rejects_windows_invalid_components(tmp_path: Path, name: str) -> None:
    package = _tar(tmp_path / "p.tar", {name: b"x"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


def test_rejects_duplicate_member_paths(tmp_path: Path) -> None:
    package = tmp_path / "p.tar"
    with tarfile.open(package, "w") as archive:
        for payload in (b"good", b"evil"):
            info = tarfile.TarInfo("a.txt")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


def test_rejects_case_folded_duplicate_member_paths(tmp_path: Path) -> None:
    package = tmp_path / "p.tar"
    with tarfile.open(package, "w") as archive:
        for name in ("A.txt", "a.txt"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("root_name", [".", "./"])
def test_skips_root_directory_member(tmp_path: Path, root_name: str) -> None:
    package = tmp_path / "p.tar"
    with tarfile.open(package, "w") as archive:
        root = tarfile.TarInfo(root_name)
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        info = tarfile.TarInfo("./a.txt")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    manifest = extract_archive(package, tmp_path / "out")
    assert [item["path"] for item in manifest["files"]] == ["a.txt"]
    assert (tmp_path / "out" / "a.txt").read_bytes() == b"x"


def test_rejects_file_member_with_empty_normalized_name(tmp_path: Path) -> None:
    package = _tar(tmp_path / "p.tar", {".": b"x"})
    with pytest.raises(ExtractionError) as caught:
        extract_archive(package, tmp_path / "out")
    assert caught.value.reason == "invalid_path"
    assert not (tmp_path / "out").exists()
