"""Results, findings, approvals, and report metadata."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eva_agent.domain.plan import Dimension, EvalPlan, SkippedDimension
from eva_agent.domain.run import RunStatus


class EvidenceRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["file", "paste_field", "dataset_row", "target_request"]
    locator: str = Field(min_length=1)
    excerpt: str | None = None


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    severity: Literal["info", "low", "medium", "high", "critical"]
    category: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    remediation: str | None = None


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    grader_calls: int = Field(default=0, ge=0)
    samples: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    duration_ms: int = Field(default=0, ge=0)


class GraderSampleOutcome(BaseModel):
    """One (grader, sample) result after score normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grader: str = Field(min_length=1)
    sample_index: int = Field(ge=0)
    score: float | None = Field(default=None, ge=0, le=1)
    raw_score: float | None = None
    reason: str = ""
    error: str | None = None


class GraderRunOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcomes: list[GraderSampleOutcome] = Field(default_factory=list)
    duration_ms: int = Field(default=0, ge=0)
    error: str | None = None


class NormalizedQuality(BaseModel):
    """Aggregated quality metrics before the final DimensionResult is built."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    metrics: dict[str, Any] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    usage: Usage


class DimensionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: Dimension
    status: Literal["ok", "partial", "skipped", "failed"]
    score: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    usage: Usage
    error: str | None = None
    truncated_reason: str | None = None


class Approval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: EvalPlan
    decided_at: datetime


class MergedResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: RunStatus
    results: list[DimensionResult] = Field(default_factory=list)
    skipped: list[SkippedDimension] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    markdown_path: str = Field(min_length=1)
    json_path: str = Field(min_length=1)
    generated_at: datetime
