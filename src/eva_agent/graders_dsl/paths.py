"""Minimal JSON path support for L2 checks and target responses."""

from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")
_MAX_PATH = 256


class PathError(ValueError):
    """Raised when a path is malformed or does not resolve."""


def parse_path(path: str) -> list[str | int]:
    if len(path) > _MAX_PATH:
        raise PathError("path exceeds 256 characters")
    if not path.startswith("$"):
        raise PathError("path must start with '$'")

    tokens: list[str | int] = []
    index = 1
    while index < len(path):
        match = _TOKEN.match(path, index)
        if match is None:
            raise PathError(f"invalid path segment at offset {index}")
        if match.group(1) is not None:
            tokens.append(match.group(1))
        else:
            tokens.append(int(match.group(2)))
        index = match.end()
    return tokens


def resolve_path(data: Any, path: str) -> Any:
    current = data
    for token in parse_path(path):
        if isinstance(token, str):
            if not isinstance(current, dict) or token not in current:
                raise PathError(f"missing key '{token}'")
            current = current[token]
        else:
            if not isinstance(current, list) or token >= len(current):
                raise PathError(f"missing index {token}")
            current = current[token]
    return current
