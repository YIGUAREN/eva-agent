"""Sample-level grading cache keyed by grader spec and sample content."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderSampleOutcome

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")


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
        folder = _SAFE_NAME.sub("_", grader)[:64] or "grader"
        return self._root / folder / f"{key[:32]}.json"

    def read(self, grader: str, key: str, sample_index: int) -> GraderSampleOutcome | None:
        path = self._path(grader, key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("cache_key") != key:
            return None
        return GraderSampleOutcome(
            grader=grader,
            sample_index=sample_index,
            score=data.get("score"),
            raw_score=data.get("raw_score"),
            reason=str(data.get("reason") or ""),
            error=data.get("error"),
        )

    def write(self, grader: str, key: str, outcome: GraderSampleOutcome) -> None:
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
