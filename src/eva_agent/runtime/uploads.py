"""Filesystem layout for uploaded evidence packages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_UPLOAD_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def validate_upload_id(value: str) -> str:
    """Reject any upload ID that could escape its directory."""
    if _UPLOAD_ID.fullmatch(value) is None:
        raise ValueError("invalid upload id")
    return value


@dataclass(frozen=True)
class UploadPaths:
    root: Path
    package: Path
    extracted: Path


def upload_paths(data_dir: str | Path, upload_id: str) -> UploadPaths:
    validate_upload_id(upload_id)
    root = Path(data_dir) / "uploads" / upload_id
    return UploadPaths(root=root, package=root / "package.bin", extracted=root / "extracted")
