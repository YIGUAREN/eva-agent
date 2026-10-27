"""Evaluation plan produced before any grading runs."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from eva_agent.domain.budget import Budget


class Dimension(StrEnum):
    QUALITY = "quality"
    LATENCY = "latency"
    COST = "cost"
    SECURITY = "security"


class DimensionPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: Dimension
    enabled: bool
    sample_size: int | None = Field(default=None, gt=0)
    notes: str = ""


class SkippedDimension(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: Dimension
    reason: str = Field(min_length=1)


class EvalPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: str = Field(min_length=1)
    evidence_summary: str
    dimensions: list[DimensionPlan] = Field(default_factory=list)
    skipped: list[SkippedDimension] = Field(default_factory=list)
    budget: Budget
