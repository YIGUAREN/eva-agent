"""Executable grader specifications stored inside an evaluation plan."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RubricSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    weight: float = Field(default=1.0, gt=0)


class CheckSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = Field(default=None, min_length=1)
    type: Literal["json_parse", "json_schema", "regex", "numeric_threshold"]
    field: str = Field(min_length=1)
    schema: dict[str, Any] | None = None  # pyright: ignore[reportIncompatibleMethodOverride]
    pattern: str | None = Field(default=None, max_length=512)
    op: Literal["<", "<=", ">", ">=", "=="] | None = None
    value: float | None = None

    @property
    def effective_name(self) -> str:
        return self.name or self.type

    @model_validator(mode="after")
    def _type_specific_fields(self) -> CheckSpec:
        if self.type == "json_parse":
            if any(value is not None for value in (self.schema, self.pattern, self.op, self.value)):
                raise ValueError("json_parse does not accept schema, pattern, op, or value")
        elif self.type == "json_schema":
            if self.schema is None:
                raise ValueError("json_schema requires schema")
            if any(value is not None for value in (self.pattern, self.op, self.value)):
                raise ValueError("json_schema does not accept pattern, op, or value")
        elif self.type == "regex":
            if self.pattern is None:
                raise ValueError("regex requires pattern")
            if any(value is not None for value in (self.schema, self.op, self.value)):
                raise ValueError("regex does not accept schema, op, or value")
        else:
            if self.op is None or self.value is None:
                raise ValueError("numeric_threshold requires op and value")
            if any(value is not None for value in (self.schema, self.pattern)):
                raise ValueError("numeric_threshold does not accept schema or pattern")
        return self


class ScoreSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    aggregation: Literal["weighted_mean"] = "weighted_mean"
    weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def _positive_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if any(weight <= 0 for weight in value.values()):
            raise ValueError("weights must be positive")
        return value


class GraderSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    level: Literal["builtin", "llm_rubric", "declarative"]
    name: str = Field(min_length=1)
    mode: Literal["pointwise", "listwise"] = "pointwise"
    input_map: dict[str, str] = Field(default_factory=dict)
    builtin_id: str | None = None
    rubric: str | None = None
    model_role: Literal["judge"] = "judge"
    checks: list[CheckSpec] | None = None
    score: ScoreSpec | None = None

    @model_validator(mode="after")
    def _declarative_fields(self) -> GraderSpec:
        if self.level == "declarative":
            if not self.checks:
                raise ValueError("declarative grader requires checks")
        else:
            if self.checks is not None:
                raise ValueError("checks are only valid for declarative graders")
            if self.score is not None:
                raise ValueError("score is only valid for declarative graders")
        return self
