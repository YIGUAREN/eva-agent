"""User-supplied run budget. No default values live in code."""

from pydantic import BaseModel, ConfigDict, Field


class Budget(BaseModel):
    """Hard limits for a single run, filled in by the user."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    max_cost_usd: float = Field(gt=0)
    max_wall_clock_s: float = Field(gt=0)
    max_target_requests: int = Field(gt=0)
