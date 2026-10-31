"""Fail-closed extraction of untrusted zip/tar evidence packages."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

_CHUNK = 1024 * 1024
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_GZIP_MAGIC = b"\x1f\x8b"
_TAR_MAGIC_OFFSET = 257
_TAR_MAGIC = b"ustar"


class ExtractionError(Exception):
    """A package violated a format or safety rule."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ExtractionLimits:
    max_file_bytes: int = 50 * 1024 * 1024
    max_total_bytes: int = 500 * 1024 * 1024
    max_files: int = 10_000
    max_depth: int = 20


def detect_format(package: Path) -> Literal["zip", "tar"]:
    """Identify zip/tar by magic bytes; extensions are ignored."""
    with package.open("rb") as handle:
        head = handle.read(512)
    if any(head.startswith(magic) for magic in _ZIP_MAGIC):
        return "zip"
    if head.startswith(_GZIP_MAGIC):
        return "tar"
    if len(head) > _TAR_MAGIC_OFFSET and head[_TAR_MAGIC_OFFSET : _TAR_MAGIC_OFFSET + len(_TAR_MAGIC)] == _TAR_MAGIC:
        return "tar"
    raise ExtractionError("unsupported_format")


def _safe_parts(name: str, limits: ExtractionLimits) -> tuple[str, ...]:
    if not name or "\x00" in name or "\\" in name:
        raise ExtractionError("invalid_path")
    normalized = name
    if normalized.startswith("/") or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ExtractionError("absolute_path")
    parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
    if not parts:
        raise ExtractionError("invalid_path")
    if any(part == ".." for part in parts):
        raise ExtractionError("path_traversal")
    if len(parts) > limits.max_depth:
        raise ExtractionError("too_deep")
    return parts


def _write_member(source, target: Path, *, limits: ExtractionLimits, total_written: int) -> tuple[int, str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with target.open("wb") as destination:
        while chunk := source.read(_CHUNK):
            size += len(chunk)
            total_written += len(chunk)
            if size > limits.max_file_bytes:
                raise ExtractionError("file_too_large")
            if total_written > limits.max_total_bytes:
                raise ExtractionError("total_too_large")
            digest.update(chunk)
            destination.write(chunk)
    return size, digest.hexdigest(), total_written


def _extract_zip(package: Path, staging: Path, limits: ExtractionLimits) -> dict:
    files: list[dict[str, object]] = []
    declared_total = 0
    total = 0
    try:
        with zipfile.ZipFile(package) as archive:
            for info in archive.infolist():
                if info.flag_bits & 0x1:
                    raise ExtractionError("encrypted")
                parts = _safe_parts(info.filename, limits)
                mode = (info.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                if info.is_dir():
                    if kind not in (0, stat.S_IFDIR):
                        raise ExtractionError("special_file")
                    continue
                if kind == stat.S_IFLNK:
                    raise ExtractionError("symlink")
                if kind not in (0, stat.S_IFREG):
                    raise ExtractionError("special_file")
                if info.file_size > limits.max_file_bytes:
                    raise ExtractionError("file_too_large")
                declared_total += info.file_size
                if declared_total > limits.max_total_bytes:
                    raise ExtractionError("total_too_large")
                if len(files) >= limits.max_files:
                    raise ExtractionError("too_many_files")
                with archive.open(info) as source:
                    size, digest, total = _write_member(
                        source, staging.joinpath(*parts), limits=limits, total_written=total
                    )
                files.append({"path": "/".join(parts), "size": size, "sha256": digest})
    except zipfile.BadZipFile as exc:
        raise ExtractionError("corrupt") from exc
    except RuntimeError as exc:
        raise ExtractionError("corrupt") from exc
    return {
        "format": "zip",
        "files": sorted(files, key=lambda item: str(item["path"])),
        "total_bytes": total,
    }


def _extract_tar(package: Path, staging: Path, limits: ExtractionLimits) -> dict:
    files: list[dict[str, object]] = []
    declared_total = 0
    total = 0
    try:
        with tarfile.open(package, mode="r:*") as archive:
            for member in archive:
                parts = _safe_parts(member.name, limits)
                if member.isdir():
                    continue
                if member.issym():
                    raise ExtractionError("symlink")
                if member.islnk():
                    raise ExtractionError("hardlink")
                if not member.isreg():
                    raise ExtractionError("special_file")
                if member.size > limits.max_file_bytes:
                    raise ExtractionError("file_too_large")
                declared_total += member.size
                if declared_total > limits.max_total_bytes:
                    raise ExtractionError("total_too_large")
                if len(files) >= limits.max_files:
                    raise ExtractionError("too_many_files")
                source = archive.extractfile(member)
                if source is None:
                    raise ExtractionError("corrupt")
                with source:
                    size, digest, total = _write_member(
                        source, staging.joinpath(*parts), limits=limits, total_written=total
                    )
                files.append({"path": "/".join(parts), "size": size, "sha256": digest})
    except tarfile.TarError as exc:
        raise ExtractionError("corrupt") from exc
    except OSError as exc:
        raise ExtractionError("corrupt") from exc
    return {
        "format": "tar",
        "files": sorted(files, key=lambda item: str(item["path"])),
        "total_bytes": total,
    }


def extract_archive(
    package: Path,
    destination: Path,
    *,
    limits: ExtractionLimits | None = None,
) -> dict:
    """Extract *package* into *destination*, replacing it only on success."""
    active_limits = limits or ExtractionLimits()
    archive_format = detect_format(package)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        if archive_format == "zip":
            manifest = _extract_zip(package, staging, active_limits)
        else:
            manifest = _extract_tar(package, staging, active_limits)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)
        raise
    shutil.rmtree(destination, ignore_errors=True)
    os.replace(staging, destination)
    return manifest
