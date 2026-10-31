"""Fail-closed extraction of untrusted zip/tar evidence packages."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import stat
import tarfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, cast
from uuid import uuid4

_CHUNK = 1024 * 1024
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_GZIP_MAGIC = b"\x1f\x8b"
_TAR_MAGIC_OFFSET = 257
_TAR_MAGIC = b"ustar"
_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD_MIN_SIZE = 22
_MAX_COMMENT_SIZE = 0xFFFF
_UNICODE_PATH_EXTRA_ID = 0x7075
_UNICODE_PATH_HEADER_SIZE = 5
_INVALID_COMPONENT_CHARS = frozenset('<>:"|?*')
# On Windows these basenames resolve to devices rather than files, even inside
# the staging directory, so writing ``NUL`` would send bytes outside the
# destination and leave a manifest attesting a file that does not exist.
_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{index}" for index in range(10)}
    | {f"LPT{index}" for index in range(10)}
)


class ExtractionError(Exception):
    """A package violated a format or safety rule."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _LimitedReader:
    """Fail closed when a single read asks for more bytes than a header may hold.

    ``tarfile`` reads a pax/GNU extended header's whole declared block in one
    call, so a tiny crafted archive can request an absurd buffer and surface as
    ``MemoryError``/``OverflowError`` (HTTP 500) instead of a rejection.
    Legitimate member data is streamed in ``_CHUNK`` reads, so a single read
    above ``max(max_file_bytes, _CHUNK)`` can only be header parsing; capping
    there keeps every valid member readable while bounding header reads.

    A cumulative budget also bounds decompression work: pax pseudo-members are
    consumed by ``tarfile`` without being yielded, so they never reach the
    per-member byte checks and could otherwise expand a tiny gzip without end.
    """

    def __init__(self, stream: IO[bytes], limit: int, *, total_limit: int | None = None) -> None:
        self._stream = stream
        self._limit = limit
        self._total_limit = total_limit
        self._consumed = 0

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0 or size > self._limit:
            # Never hand ``read()``-everything through to the stream.
            size = self._limit
        data = self._stream.read(size)
        self._consumed += len(data)
        if self._total_limit is not None and self._consumed > self._total_limit:
            raise ExtractionError("total_too_large")
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def tell(self) -> int:
        return self._stream.tell()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


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


def _reject_unsafe_component(component: str) -> None:
    """Reject a single path component that is invalid or unsafe on Windows."""
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in component):
        raise ExtractionError("invalid_path")
    if any(char in _INVALID_COMPONENT_CHARS for char in component):
        raise ExtractionError("invalid_path")
    if component.endswith((".", " ")):
        raise ExtractionError("invalid_path")
    basename = component.split(".", 1)[0].rstrip(" .").upper()
    if basename in _RESERVED_DEVICE_NAMES:
        raise ExtractionError("invalid_path")


def _safe_parts(name: str, limits: ExtractionLimits, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not name or "\x00" in name or "\\" in name:
        raise ExtractionError("invalid_path")
    normalized = name
    if normalized.startswith("/") or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ExtractionError("absolute_path")
    parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
    if not parts:
        if allow_empty:
            return ()
        raise ExtractionError("invalid_path")
    if any(part == ".." for part in parts):
        raise ExtractionError("path_traversal")
    for component in parts:
        _reject_unsafe_component(component)
    if len(parts) > limits.max_depth:
        raise ExtractionError("too_deep")
    return parts


def _reject_unsafe_unicode_path_extra(info: zipfile.ZipInfo) -> None:
    """Reject a 0x7075 Unicode-path extra field that hides an unsafe name.

    ``zipfile`` replaces ``ZipInfo.filename`` with the 0x7075 payload and then
    sanitizes it, so a NUL truncates or a backslash is rewritten before the
    effective-name checks run. The raw payload is only available through
    ``ZipInfo.extra`` and must be validated with the same NUL/backslash rules
    as a raw member name.
    """
    extra = info.extra
    offset = 0
    while offset + 4 <= len(extra):
        header_id = int.from_bytes(extra[offset : offset + 2], "little")
        data_size = int.from_bytes(extra[offset + 2 : offset + 4], "little")
        data_start = offset + 4
        data_end = data_start + data_size
        if data_end > len(extra):
            # Cannot skip past the declared field; fail closed rather than
            # lose track of a later 0x7075 payload.
            raise ExtractionError("invalid_path")
        if header_id == _UNICODE_PATH_EXTRA_ID:
            data = extra[data_start:data_end]
            if len(data) < _UNICODE_PATH_HEADER_SIZE:
                raise ExtractionError("invalid_path")
            try:
                unicode_path = data[_UNICODE_PATH_HEADER_SIZE:].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ExtractionError("invalid_path") from exc
            if "\x00" in unicode_path or "\\" in unicode_path:
                raise ExtractionError("invalid_path")
        offset = data_end


def _reject_multi_volume(package: Path) -> tuple[int, bytes]:
    """Reject a zip whose end-of-central-directory declares another disk.

    Like ``zipfile``, only the last EOCD signature inside the window a valid
    archive comment could occupy is considered; earlier occurrences inside
    stored member data are inert. That last candidate must end exactly at EOF
    (its declared comment length reaches the end of the file) and declare
    disk 0 for both the archive and the central directory. A missing or
    malformed candidate fails closed as ``unsupported_format``, so appended
    bytes cannot desynchronize this check from ``zipfile``.

    Returns the EOCD offset within the returned tail so callers can bound the
    declared entry count and central-directory size without re-reading.
    """
    with package.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        window = min(file_size, _EOCD_MIN_SIZE + _MAX_COMMENT_SIZE)
        handle.seek(file_size - window)
        tail = handle.read()
    start = tail.rfind(_EOCD_SIGNATURE)
    if start < 0:
        raise ExtractionError("unsupported_format")
    if start + _EOCD_MIN_SIZE > len(tail):
        raise ExtractionError("unsupported_format")
    comment_length = int.from_bytes(tail[start + 20 : start + 22], "little")
    if start + _EOCD_MIN_SIZE + comment_length != len(tail):
        raise ExtractionError("unsupported_format")
    disk_number = int.from_bytes(tail[start + 4 : start + 6], "little")
    central_disk = int.from_bytes(tail[start + 6 : start + 8], "little")
    if disk_number != 0 or central_disk != 0:
        raise ExtractionError("unsupported_format")
    return start, tail


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
    seen: set[str] = set()
    declared_total = 0
    total = 0
    try:
        eocd_start, eocd_tail = _reject_multi_volume(package)
        # Bound the central directory before ``ZipFile`` materialises one
        # ``ZipInfo`` per entry: a small archive can declare ~1M entries.
        declared_entries = int.from_bytes(eocd_tail[eocd_start + 10 : eocd_start + 12], "little")
        central_directory_size = int.from_bytes(eocd_tail[eocd_start + 12 : eocd_start + 16], "little")
        # 0xFFFF is the ZIP64 "count is elsewhere" marker, so it cannot be
        # compared numerically. It only implies an over-limit count while the
        # configured limit is below 65,535; above that, the per-member counter
        # below remains authoritative.
        if declared_entries == 0xFFFF and limits.max_files < 0xFFFF:
            raise ExtractionError("too_many_files")
        if declared_entries != 0xFFFF and declared_entries > limits.max_files:
            raise ExtractionError("too_many_files")
        if central_directory_size > max(limits.max_files * 1024, 1024 * 1024):
            raise ExtractionError("too_many_files")
        with zipfile.ZipFile(package) as archive:
            member_count = 0
            for info in archive.infolist():
                member_count += 1
                # Directory entries are members too and consume the same budget,
                # so a directory-heavy archive cannot bypass the member limit.
                if member_count > limits.max_files:
                    raise ExtractionError("too_many_files")
                if info.flag_bits & 0x1:
                    raise ExtractionError("encrypted")
                # ``ZipInfo.filename`` is already sanitized (NUL truncated,
                # backslashes rewritten on Windows); the declared name is only
                # available through ``orig_filename``.
                if "\x00" in info.orig_filename or "\\" in info.orig_filename:
                    raise ExtractionError("invalid_path")
                _reject_unsafe_unicode_path_extra(info)
                if info.volume != 0:
                    raise ExtractionError("unsupported_format")
                is_dir = info.is_dir()
                parts = _safe_parts(info.filename, limits, allow_empty=is_dir)
                mode = (info.external_attr >> 16) & 0xFFFF
                kind = stat.S_IFMT(mode)
                if is_dir:
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
                manifest_path = "/".join(parts)
                folded_path = manifest_path.casefold()
                if folded_path in seen:
                    raise ExtractionError("invalid_path")
                seen.add(folded_path)
                with archive.open(info) as source:
                    size, digest, total = _write_member(
                        source, staging.joinpath(*parts), limits=limits, total_written=total
                    )
                files.append({"path": manifest_path, "size": size, "sha256": digest})
        if not files:
            raise ExtractionError("rejected")
    except (MemoryError, OverflowError) as exc:
        raise ExtractionError("corrupt") from exc
    except zipfile.BadZipFile as exc:
        raise ExtractionError("corrupt") from exc
    except RuntimeError as exc:
        raise ExtractionError("corrupt") from exc
    except zlib.error as exc:
        raise ExtractionError("corrupt") from exc
    except OSError as exc:
        raise ExtractionError("invalid_path") from exc
    return {
        "format": "zip",
        "files": sorted(files, key=lambda item: str(item["path"])),
        "total_bytes": total,
    }


def _extract_tar(package: Path, staging: Path, limits: ExtractionLimits) -> dict:
    files: list[dict[str, object]] = []
    seen: set[str] = set()
    declared_total = 0
    total = 0
    try:
        with package.open("rb") as raw:
            is_gzip = raw.read(len(_GZIP_MAGIC)) == _GZIP_MAGIC
            raw.seek(0)
            # Wrap the decompressed stream so the header-read cap applies to
            # what ``tarfile`` actually parses (plain or gzip, the only formats
            # ``detect_format`` accepts).
            stream = cast(IO[bytes], gzip.GzipFile(fileobj=raw) if is_gzip else raw)
            guarded = _LimitedReader(
                stream,
                max(limits.max_file_bytes, _CHUNK),
                # Legitimate member bytes are bounded by max_total_bytes; the
                # extra headroom covers headers and any re-read by tarfile.
                total_limit=2 * limits.max_total_bytes + limits.max_file_bytes,
            )
            with tarfile.open(fileobj=cast(IO[bytes], guarded), mode="r:") as archive:
                member_count = 0
                for member in archive:
                    member_count += 1
                    # Directory entries are members too; count them before skipping.
                    if member_count > limits.max_files:
                        raise ExtractionError("too_many_files")
                    if member.isdir():
                        # A root entry named "." / "./" (``tar -cf pkg.tar .``)
                        # normalizes to no parts and is a harmless no-op.
                        _safe_parts(member.name, limits, allow_empty=True)
                        continue
                    parts = _safe_parts(member.name, limits)
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
                    manifest_path = "/".join(parts)
                    folded_path = manifest_path.casefold()
                    if folded_path in seen:
                        raise ExtractionError("invalid_path")
                    seen.add(folded_path)
                    source = archive.extractfile(member)
                    if source is None:
                        raise ExtractionError("corrupt")
                    with source:
                        size, digest, total = _write_member(
                            source, staging.joinpath(*parts), limits=limits, total_written=total
                        )
                    files.append({"path": manifest_path, "size": size, "sha256": digest})
        if not files:
            raise ExtractionError("rejected")
    except (MemoryError, OverflowError) as exc:
        raise ExtractionError("corrupt") from exc
    except (tarfile.TarError, EOFError, zlib.error) as exc:
        # A truncated gzip stream surfaces as EOFError from ``gzip`` rather
        # than TarError; treat it as a corrupt package, not a server failure.
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
    """Extract *package* into *destination*.

    Extraction is staged in a temporary sibling directory and moved into place
    on success. On any failure the staging directory and *destination* are both
    removed, so a pre-existing destination is not preserved (fail closed).
    """
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
    try:
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest
