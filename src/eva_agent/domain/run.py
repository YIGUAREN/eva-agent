"""Run input and lifecycle types."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputKind(StrEnum):
    UPLOAD = "upload"
    PASTE = "paste"
    DATASET = "dataset"
    TARGET = "target"


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


class RunInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: InputKind
    paste: PastedInput | None = None
    upload_id: str | None = None
    dataset_id: str | None = None
    target_ref: str | None = None

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> RunInput:
        if self.kind is InputKind.PASTE and self.paste is None:
            raise ValueError("paste input requires 'paste'")
        if self.kind is InputKind.UPLOAD and not self.upload_id:
            raise ValueError("upload input requires 'upload_id'")
        if self.kind is InputKind.DATASET and not self.dataset_id:
            raise ValueError("dataset input requires 'dataset_id'")
        if self.kind is InputKind.TARGET and not self.target_ref:
            raise ValueError("target input requires 'target_ref'")
        return self
