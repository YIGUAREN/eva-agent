"""Sample-level grading cache keyed by grader spec and sample content."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderSampleOutcome

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
)


def _safe_folder(grader: str) -> str:
    folder = _SAFE_NAME.sub("_", grader)[:64] or "grader"
    # Windows resolves CON/NUL/COM1… as devices even with an extension; a
    # leading underscore keeps the cache path an ordinary directory.
    return f"_{folder}" if folder.upper() in _RESERVED_NAMES else folder


def cache_key(spec: GraderSpec, row: Mapping[str, Any]) -> str:
    material = json.dumps(
        {"spec": spec.model_dump(mode="json"), "row": row},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class GradingCache:
    def __init__(self, root: Path, run_id: str) -> None:
        self._root = root / "runs" / run_id / "gradings"

    def _path(self, grader: str, key: str) -> Path:
        return self._root / _safe_folder(grader) / f"{key[:32]}.json"

    def read(self, grader: str, key: str, sample_index: int) -> GraderSampleOutcome | None:
        path = self._path(grader, key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("cache_key") != key:
                return None
            score = data.get("score")
            error = data.get("error")
            # A record with neither value carries no grading result; re-grade
            # instead of serving an empty hit forever.
            if score is None and error is None:
                return None
            return GraderSampleOutcome(
                grader=grader,
                sample_index=sample_index,
                score=score,
                raw_score=data.get("raw_score"),
                reason=str(data.get("reason") or ""),
                error=error,
            )
        except (OSError, ValueError):
            # A validation failure is a ValueError; any unreadable or malformed
            # record is a cache miss, never a run failure.
            return None

    def write(self, grader: str, key: str, outcome: GraderSampleOutcome) -> None:
        # The model call is already paid for; a cache write is best-effort.
        try:
            path = self._path(grader, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cache_key": key,
                "score": outcome.score,
                "raw_score": outcome.raw_score,
                "reason": outcome.reason,
                "error": outcome.error,
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("could not write grading cache for grader %r", grader, exc_info=True)
