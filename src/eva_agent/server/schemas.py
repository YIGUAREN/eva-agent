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


def public_run_detail(row: RunRow, *, internal_hosts: tuple[str, ...] = ()) -> dict[str, Any]:
    summary = run_summary(row)
    objective = summary.get("objective")
    if isinstance(objective, str):
        # An objective that carries any absolute path is replaced wholesale: a
        # token-level regex cannot distinguish the tail of a spaced path or a
        # UNC share from surrounding prose.
        if _has_absolute_path(objective):
            summary["objective"] = "<path>"
        else:
            summary["objective"] = redact_internal_addresses_and_secrets(
                objective, internal_hosts=internal_hosts
            )
    return summary


# Characters that end an absolute-path token. The CJK punctuation set keeps
# paths quoted inside Chinese prose intact; ``#`` and ``?`` stay outside the
# token so line references (``#3``) and query strings survive redaction.
_PATH_TERMINATORS = "\\s`\"'<>，。；：、（）【】#?"

_PATH_CHARS = rf"[^{_PATH_TERMINATORS}]"

# A path body. A space is consumed only when the rest of the same token still
# contains a path separator, so ``C:\Users\John Doe\ds.jsonl`` is redacted as a
# unit while prose such as ``run at D:\data now`` stops at the word boundary.
_PATH_BODY = rf"(?:{_PATH_CHARS}| (?={_PATH_CHARS}*[\\/]))+"

# Windows drive-letter paths. The lookbehind blocks the "p:/" tail of URLs such
# as http:// and keeps the drive letter at a token boundary.
_WINDOWS_ABSOLUTE_PATH = re.compile(rf"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{_PATH_BODY}")

# Windows UNC paths (``\\server\share\file``). Backslashes are not path
# terminators, so the whole share token is consumed.
_WINDOWS_UNC_PATH = re.compile(rf"(?<![A-Za-z0-9])\\\\{_PATH_BODY}")

# POSIX UNC paths (``//server/share/file``). The lookbehind keeps the ``//`` in
# ``http://`` and ``https://`` from being mistaken for a network share.
_POSIX_UNC_PATH = re.compile(rf"(?<![:/])//{_PATH_BODY}")

# POSIX absolute paths starting at a token boundary. The lookbehind rejects
# ASCII word characters, a colon (http://), a slash (//host) and a hyphen (so
# GitLab URLs such as ``-/jobs/1`` survive), while still allowing a backtick,
# a quote, whitespace or CJK punctuation right before the leading slash.
_POSIX_ABSOLUTE_PATH = re.compile(
    rf"(?<![A-Za-z0-9_:/\\-])/[^{_PATH_TERMINATORS}/](?:{_PATH_BODY})?"
)

_ABSOLUTE_PATH_PATTERNS = (
    _WINDOWS_ABSOLUTE_PATH,
    _WINDOWS_UNC_PATH,
    _POSIX_UNC_PATH,
    _POSIX_ABSOLUTE_PATH,
)

# Run-relative AIG artifact names, e.g. data/runs/<id>/aig/<session>.json or aig/<session>.csv.
# They are admin-only artifacts, so public output must never carry them.
_AIG_ARTIFACT_PATH = re.compile(
    r"(?<![A-Za-z0-9._-])(?:[A-Za-z0-9._-]+[\\/])*(?i:aig)[\\/][A-Za-z0-9_-]{1,128}\.(?:json|csv)"
    r"(?![A-Za-z0-9._-])"
)
_PUBLIC_PATH_PATTERNS = (*_ABSOLUTE_PATH_PATTERNS, _AIG_ARTIFACT_PATH)


def _has_absolute_path(text: str) -> bool:
    """Return True when *text* contains any absolute or run-relative AIG artifact path token."""
    return any(pattern.search(text) is not None for pattern in _PUBLIC_PATH_PATTERNS)


def redact_absolute_paths(text: str) -> str:
    """Replace absolute or run-relative AIG artifact path tokens with ``<path>``.

    Line references such as ``D:\\data\\ds.jsonl#3`` become ``<path>#3``.
    """
    for pattern in _PUBLIC_PATH_PATTERNS:
        text = pattern.sub("<path>", text)
    return text


def redact_markdown_paths(text: str) -> str:
    """Redact absolute or run-relative AIG artifact path markdown lines.

    The absolute-path token regex stops at a space unless a later separator
    proves the space belongs to the path, so a line ending in a spaced
    component or a UNC share would leak its tail (``<path> File.jsonl``). Per
    line, this keeps the text before the first path and the ``#<digits>`` line
    reference, then drops the remainder. Over-redacting the rest of the line is
    intentional. Run-relative AIG artifact names are exact, space-free tokens,
    so they are replaced in place without dropping the rest of the line.
    """
    redacted: list[str] = []
    for line in text.split("\n"):
        line = _AIG_ARTIFACT_PATH.sub("<path>", line)
        matches = [match for pattern in _PUBLIC_PATH_PATTERNS if (match := pattern.search(line))]
        if not matches:
            redacted.append(line)
            continue
        first = min(matches, key=lambda match: match.start())
        reference = re.match(r"#\d+", line[first.end() :])
        suffix = reference.group(0) if reference else ""
        redacted.append(f"{line[:first.start()]}<path>{suffix}")
    return "\n".join(redacted)


# Loopback and RFC1918 addresses, with an optional port. The lookarounds keep
# the match from starting or ending inside a longer dotted number.
_PRIVATE_IPV4 = re.compile(
    r"(?<![\d.])"
    r"(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|127\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})"
    r"(?::\d{1,5})?(?!\d)"
)

_LOCALHOST = re.compile(r"(?<![A-Za-z0-9_.-])localhost(?::\d{1,5})?(?![A-Za-z0-9_.-])", re.IGNORECASE)

_IPV6_LOOPBACK = re.compile(r"(?<![0-9A-Za-z:])\[?::1\]?(?::\d{1,5})?(?![0-9A-Za-z:])")

_SK_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")

_BEARER_TOKEN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)

_ENV_SECRET = re.compile(
    r"(?P<key>\b(?:api_key|access_token|auth_token|token|secret|password))"
    r"(?P<sep>\s*[=:]\s*)"
    r"(?P<value>[^\s\"',;&]{6,})",
    re.IGNORECASE,
)


def redact_internal_addresses_and_secrets(text: str, *, internal_hosts: tuple[str, ...] = ()) -> str:
    """Replace internal addresses and credential-shaped tokens in free text.

    Private/loopback addresses, configured internal hostnames, ``sk-`` tokens,
    bearer tokens, and env-style secret assignments never reach the public API.
    """
    text = _SK_TOKEN.sub("<secret>", text)
    text = _BEARER_TOKEN.sub("<secret>", text)
    text = _ENV_SECRET.sub(lambda match: f"{match.group('key')}{match.group('sep')}<secret>", text)
    text = _PRIVATE_IPV4.sub("<internal>", text)
    text = _LOCALHOST.sub("<internal>", text)
    text = _IPV6_LOOPBACK.sub("<internal>", text)
    hosts = tuple(host for host in (item.strip() for item in internal_hosts) if host)
    if hosts:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_.-])(?:"
            + "|".join(re.escape(host) for host in hosts)
            + r")(?::\d{1,5})?(?![A-Za-z0-9_.-])",
            re.IGNORECASE,
        )
        text = pattern.sub("<internal>", text)
    return text


def sanitize_public_text(text: str, *, internal_hosts: tuple[str, ...] = ()) -> str:
    """Sanitize free text (an objective, for example) for anonymous readers."""
    return redact_internal_addresses_and_secrets(redact_absolute_paths(text), internal_hosts=internal_hosts)


def _is_sensitive_key(key: str) -> bool:
    """Return True when a payload key must never reach the public API."""
    return key in _SENSITIVE_KEYS or key.endswith("_path")


def sanitize_public_payload(value: Any, *, internal_hosts: tuple[str, ...] = ()) -> Any:
    """Recursively sanitize a payload before it reaches the public API.

    Sensitive keys are dropped. A string value that contains any absolute or
    run-relative AIG artifact path token (a Windows drive path, a Windows or
    POSIX UNC share, a POSIX absolute path, or ``…/aig/<id>.json|csv``) is
    replaced entirely with ``<path>``; whole-value replacement prevents paths
    containing spaces or UNC shares from leaking their tail.
    Other strings have internal addresses and secrets redacted.
    """
    if isinstance(value, dict):
        return {
            key: sanitize_public_payload(item, internal_hosts=internal_hosts)
            for key, item in value.items()
            if not _is_sensitive_key(key)
        }
    if isinstance(value, list):
        return [sanitize_public_payload(item, internal_hosts=internal_hosts) for item in value]
    if isinstance(value, str):
        if _has_absolute_path(value):
            return "<path>"
        return redact_internal_addresses_and_secrets(value, internal_hosts=internal_hosts)
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
