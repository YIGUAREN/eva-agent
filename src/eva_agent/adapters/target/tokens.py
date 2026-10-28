"""Local token estimation used only when the target omits usage."""

from __future__ import annotations

from functools import lru_cache

import tiktoken


class TokenEstimationError(ValueError):
    """Raised when the requested encoding is unavailable."""


@lru_cache(maxsize=8)
def _encoding(name: str):
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # tiktoken raises ValueError or KeyError depending on version
        raise TokenEstimationError(f"token estimation failed for encoding {name!r}: {exc}") from exc


def estimate_tokens(text: str, encoding: str) -> int:
    return len(_encoding(encoding).encode(text))
