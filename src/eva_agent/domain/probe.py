"""Records produced by one target-probe request."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProbeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    prompt: str
    started_at: datetime
    status_code: int | None = None
    latency_ms: int = Field(ge=0)
    response_text: str = ""
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    usage_source: Literal["target", "estimated", "missing"] = "missing"
    error: str | None = None


class ProbeSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
