"""Dataset rows prepared for one expert run."""

from pydantic import BaseModel, ConfigDict, Field


class DatasetSample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, str]] = Field(default_factory=list)
