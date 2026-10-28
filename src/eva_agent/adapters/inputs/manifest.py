"""Deterministic evidence manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eva_agent.domain.evidence import EvidenceFile
from eva_agent.domain.run import DatasetRef, InputKind
from eva_agent.domain.target import TargetSpec


def canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(
    evidence_dir: Path,
    *,
    kind: InputKind,
    files: list[EvidenceFile],
    pasted: dict[str, str] | None,
    dataset: DatasetRef | None,
    target: TargetSpec | None = None,
    target_prompts: list[str] | None = None,
) -> str:
    """Write manifest.json and return the sha256 of its canonical form."""
    payload = {
        "kind": kind.value,
        "files": [item.model_dump(mode="json") for item in files],
        "pasted": pasted,
        "dataset": dataset.model_dump(mode="json") if dataset is not None else None,
        "target": target.model_dump(mode="json") if target is not None else None,
        "target_prompts": list(target_prompts or []),
    }
    canonical = canonical_json(payload)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "manifest.json").write_bytes(canonical)
    return sha256_bytes(canonical)
