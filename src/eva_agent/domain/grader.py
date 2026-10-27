"""Executable grader specifications stored inside an evaluation plan."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RubricSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    weight: float = Field(default=1.0, gt=0)


class GraderSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    level: Literal["builtin", "llm_rubric", "declarative"]
    name: str = Field(min_length=1)
    mode: Literal["pointwise", "listwise"] = "pointwise"
    input_map: dict[str, str] = Field(default_factory=dict)
    builtin_id: str | None = None
    rubric: str | None = None
    model_role: Literal["judge"] = "judge"
