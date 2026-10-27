"""Dataset rows prepared for one expert run."""

from pydantic import BaseModel, ConfigDict, Field


class DatasetSample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # BaseModel.schema() is deprecated but still declared; the field name is required by the domain interface.
    schema: list[str] = Field(default_factory=list)  # pyright: ignore[reportIncompatibleMethodOverride]
    rows: list[dict[str, str]] = Field(default_factory=list)
