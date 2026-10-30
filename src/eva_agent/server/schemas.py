"""Request and response models plus safe run serializers."""

from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from eva_agent.domain.budget import Budget
from eva_agent.domain.run import RunInput
from eva_agent.server.errors import ApiError
from eva_agent.storage.models import RunRow, ensure_utc

_SENSITIVE_KEYS = frozenset(
    {
        "markdown_path",
        "json_path",
        "path",
        "output_path",
        "report_json_path",
        "input_json",
        "budget",
        "plan",
    }
)


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: RunInput
    budget: Budget
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)
    authorization_ref: str | None = Field(default=None, max_length=256)


class PublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    published: bool


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    plan: dict[str, Any] | None = None


def _safe_input(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    dataset = data.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("path"), str):
        data["dataset"] = {**dataset, "path": Path(dataset["path"]).name}
    return data


def run_summary(row: RunRow) -> dict[str, Any]:
    return {
        "run_id": row.id,
        "status": row.status,
        "objective": (row.input_json or {}).get("objective"),
        "created_at": ensure_utc(row.created_at).isoformat(),
        "published_at": ensure_utc(row.published_at).isoformat() if row.published_at else None,
    }


def run_detail(row: RunRow) -> dict[str, Any]:
    return {
        **run_summary(row),
        "input": _safe_input(row.input_json),
        "budget": row.budget_json,
        "plan": row.plan_json,
        "error": row.error,
        "is_public": row.is_public,
        "created_by": row.created_by,
    }


def public_run_detail(row: RunRow) -> dict[str, Any]:
    return run_summary(row)


# Characters that end an absolute-path token. The CJK punctuation set keeps
# paths quoted inside Chinese prose intact; ``#`` and ``?`` stay outside the
# token so line references (``#3``) and query strings survive redaction.
_PATH_TERMINATORS = "\\s`\"'<>，。；：、（）【】#?"

# Windows drive-letter paths. The lookbehind blocks the "p:/" tail of URLs such
# as http:// and keeps the drive letter at a token boundary.
_WINDOWS_ABSOLUTE_PATH = re.compile(rf"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^{_PATH_TERMINATORS}]*")

# Windows UNC paths (``\\server\share\file``). Backslashes are not path
# terminators, so the whole share token is consumed.
_WINDOWS_UNC_PATH = re.compile(rf"(?<![A-Za-z0-9])\\\\[^{_PATH_TERMINATORS}]+")

# POSIX UNC paths (``//server/share/file``). The lookbehind keeps the ``//`` in
# ``http://`` and ``https://`` from being mistaken for a network share.
_POSIX_UNC_PATH = re.compile(rf"(?<![:/])//[^{_PATH_TERMINATORS}]+")

# POSIX absolute paths starting at a token boundary. The lookbehind rejects
# ASCII word characters, a colon (http://), a slash (//host) and a hyphen (so
# GitLab URLs such as ``-/jobs/1`` survive), while still allowing a backtick,
# a quote, whitespace or CJK punctuation right before the leading slash.
_POSIX_ABSOLUTE_PATH = re.compile(rf"(?<![A-Za-z0-9_:/\\-])/[^{_PATH_TERMINATORS}/][^{_PATH_TERMINATORS}]*")

_ABSOLUTE_PATH_PATTERNS = (
    _WINDOWS_ABSOLUTE_PATH,
    _WINDOWS_UNC_PATH,
    _POSIX_UNC_PATH,
    _POSIX_ABSOLUTE_PATH,
)


def _has_absolute_path(text: str) -> bool:
    """Return True when *text* contains any recognized absolute path token."""
    return any(pattern.search(text) is not None for pattern in _ABSOLUTE_PATH_PATTERNS)


def redact_absolute_paths(text: str) -> str:
    """Replace absolute filesystem path tokens with ``<path>``.

    Line references such as ``D:\\data\\ds.jsonl#3`` become ``<path>#3``.
    """
    for pattern in _ABSOLUTE_PATH_PATTERNS:
        text = pattern.sub("<path>", text)
    return text


def _is_sensitive_key(key: str) -> bool:
    """Return True when a payload key must never reach the public API."""
    return key in _SENSITIVE_KEYS or key.endswith("_path")


def sanitize_public_payload(value: Any) -> Any:
    """Recursively sanitize a payload before it reaches the public API.

    Sensitive keys are dropped. A string value that contains any absolute path
    token (a Windows drive path, a Windows or POSIX UNC share, or a POSIX
    absolute path) is replaced entirely with ``<path>``; whole-value replacement
    prevents paths containing spaces or UNC shares from leaking their tail.
    Other strings are returned unchanged.
    """
    if isinstance(value, dict):
        return {key: sanitize_public_payload(item) for key, item in value.items() if not _is_sensitive_key(key)}
    if isinstance(value, list):
        return [sanitize_public_payload(item) for item in value]
    if isinstance(value, str):
        return "<path>" if _has_absolute_path(value) else value
    return value


def encode_cursor(row: RunRow) -> str:
    raw = f"{ensure_utc(row.created_at).isoformat()}|{row.id}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def decode_cursor(value: str) -> tuple[datetime, str]:
    try:
        raw = base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8")
        stamp, _, run_id = raw.partition("|")
        if not run_id:
            raise ValueError("missing run id")
        return datetime.fromisoformat(stamp), run_id
    except (ValueError, UnicodeDecodeError, binascii.Error):
        raise ApiError(422, "invalid_cursor", "invalid cursor") from None
