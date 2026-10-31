"""Attachment discovery and download limits for AIG model-redteam results."""

from __future__ import annotations

import re
from typing import Any

DOWNLOAD_PATH = "/api/v1/app/tasks/{session_id}/downloadFile"
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
# The live result carries one attachment; two distinct names covers the observed
# naming while keeping the tool budget honest.
MAX_ATTACHMENTS = 2

ATTACHMENT_NAME = re.compile(r"(?!\.)[A-Za-z0-9._-]{1,255}\.csv")


def attachment_names(payload: dict[str, Any]) -> list[str]:
    """Return the CSV attachment names AIG reported for this task result."""
    result = payload.get("result")
    content = result.get("content") if isinstance(result, dict) else None
    if not isinstance(content, list):
        return []
    names: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        name = item.get("attachment")
        if isinstance(name, str) and ATTACHMENT_NAME.fullmatch(name):
            names.append(name)
    return names
