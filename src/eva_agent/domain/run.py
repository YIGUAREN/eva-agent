"""Run input and lifecycle types."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


class InputKind(StrEnum):
    UPLOAD = "upload"
    PASTE = "paste"
    DATASET = "dataset"
    TARGET = "target"


def validate_run_id(run_id: str) -> str:
    """Reject run identifiers that could escape the per-run data directory."""
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(f"invalid run_id: {run_id[:32]!r}")
    return run_id


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_BUDGET = "waiting_budget"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PastedInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    response: str | None = None
    context: str | None = None
    transcript: str | None = None


class DatasetRef(BaseModel):
    """A dataset file and how its columns map to canonical fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    format: Literal["csv", "jsonl"]
    column_map: dict[str, str] = Field(default_factory=dict)


class RunInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: InputKind
    objective: str | None = None
    paste: PastedInput | None = None
    dataset: DatasetRef | None = None
    upload_id: str | None = None
    dataset_id: str | None = None
    target_ref: str | None = None

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> RunInput:
        if self.kind is InputKind.PASTE and self.paste is None:
            raise ValueError("paste input requires 'paste'")
        if self.kind is InputKind.UPLOAD and not self.upload_id:
            raise ValueError("upload input requires 'upload_id'")
        if self.kind is InputKind.DATASET and self.dataset is None and not self.dataset_id:
            raise ValueError("dataset input requires 'dataset' or 'dataset_id'")
        if self.kind is InputKind.TARGET and not self.target_ref:
            raise ValueError("target input requires 'target_ref'")
        if self.kind is not InputKind.PASTE and self.paste is not None:
            raise ValueError("'paste' is only valid for paste input")
        if self.kind is not InputKind.DATASET and self.dataset is not None:
            raise ValueError("'dataset' is only valid for dataset input")
        return self
