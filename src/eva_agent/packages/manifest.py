"""Integrity-checked package manifests shared by grader packages and run bundles."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eva_agent.sandbox.extract import ExtractionError, safe_member_parts

MANIFEST_NAME = "manifest.json"
MAX_PACKAGE_BYTES = 25 * 1024 * 1024
MAX_PACKAGE_FILES = 512
_SYMLINK_MODE = 0o120000


class PackageError(ValueError):
    """A package failed its integrity or layout check."""


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(min_length=64, max_length=64)
    size: int = Field(ge=0)


class PackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["grader_package", "run_bundle"]
    format_version: int = 1
    run_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    report_version: int | None = None
    catalog_sha256: str | None = None
    evidence_sha256: str | None = None
    replayable: bool = True
    replay_blockers: list[str] = Field(default_factory=list)
    files: dict[str, FileEntry]


def write_manifest(root: Path, manifest: PackageManifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_NAME
    path.write_bytes(canonical_json(manifest.model_dump(mode="json")))
    return path


def read_manifest(root: Path) -> PackageManifest:
    path = root / MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"cannot read {MANIFEST_NAME}: {exc}") from None
    try:
        return PackageManifest.model_validate(raw)
    except ValidationError as exc:
        raise PackageError(f"invalid {MANIFEST_NAME}: {exc}") from None


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def verify_tree(root: Path, manifest: PackageManifest) -> None:
    """Every listed member exists with its recorded hash and nothing else is present."""
    for name in manifest.files:
        _member_name(name)
    for name in manifest.files:
        path = root / name
        if path.exists() and not path.is_file():
            raise PackageError(f"member is not a file: {name}")
    expected = set(manifest.files)
    actual = _relative_files(root)
    actual.discard(MANIFEST_NAME)
    missing = sorted(expected - actual)
    if missing:
        raise PackageError(f"package is missing {missing}")
    extra = sorted(actual - expected)
    if extra:
        raise PackageError(f"package contains unexpected files {extra}")
    for name in sorted(expected):
        entry = manifest.files[name]
        path = root / name
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise PackageError(f"cannot read member {name}: {exc}") from None
        if size != entry.size:
            raise PackageError(f"size mismatch for {name}")
        try:
            digest = sha256_file(path)
        except OSError as exc:
            raise PackageError(f"cannot read member {name}: {exc}") from None
        if digest != entry.sha256:
            raise PackageError(f"hash mismatch for {name}")


def _member_name(name: str) -> str:
    try:
        parts = safe_member_parts(name)
    except ExtractionError as exc:
        raise PackageError(f"invalid member name {name!r}: {exc.reason}") from None
    return "/".join(parts)


def write_zip(root: Path, destination: Path, *, manifest: PackageManifest) -> Path:
    names = [_member_name(name) for name in (MANIFEST_NAME, *sorted(manifest.files))]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, arcname=name)
    return destination


def extract_zip(source: Path, destination: Path) -> PackageManifest:
    """Extract a package into ``destination`` with layout limits, then verify its hashes."""
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackageError(f"not a zip package: {exc}") from None
    with archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > MAX_PACKAGE_FILES:
            raise PackageError(f"package has {len(members)} files; the limit is {MAX_PACKAGE_FILES}")
        seen: set[str] = set()
        total = 0
        names: list[str] = []
        for info in members:
            name = _member_name(info.filename)
            if name in seen:
                raise PackageError(f"duplicate package member {name}")
            seen.add(name)
            if (info.external_attr >> 16) & 0o170000 == _SYMLINK_MODE:
                raise PackageError(f"package member is a symlink: {name}")
            total += info.file_size
            if total > MAX_PACKAGE_BYTES:
                raise PackageError(f"package expands past {MAX_PACKAGE_BYTES} bytes")
            names.append(name)
        destination.mkdir(parents=True, exist_ok=True)
        written = 0
        for info, name in zip(members, names, strict=True):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as reader, target.open("wb") as writer:
                while chunk := reader.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_PACKAGE_BYTES:
                        raise PackageError(f"package expands past {MAX_PACKAGE_BYTES} bytes")
                    writer.write(chunk)
    manifest = read_manifest(destination)
    verify_tree(destination, manifest)
    return manifest
