"""Validated AIG model-redteam result models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from eva_agent.adapters.aig.errors import AigResultError


class AigVulnerability(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)

    vulnerability: str = Field(min_length=1)
    total: int = Field(ge=0)
    jailbreak: int = Field(ge=0)
    score: float | None = Field(default=None, ge=0, le=100)
    asr: float | None = Field(default=None, ge=0, le=1)
    errored: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _counts_fit(self) -> AigVulnerability:
        if self.jailbreak > self.total:
            raise ValueError("jailbreak exceeds total")
        return self


class AigAttackMethod(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)

    attack_method: str = Field(alias="attackMethod", min_length=1)
    total: int = Field(ge=0)
    jailbreak: int = Field(ge=0)
    score: float | None = Field(default=None, ge=0, le=100)
    asr: float | None = Field(default=None, ge=0, le=1)
    errored: int = Field(default=0, ge=0)


class AigExtraBody(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    vulnerability_results: list[AigVulnerability] = Field(
        default_factory=list, alias="vulnerabilityResults"
    )
    attack_method_results: list[AigAttackMethod] = Field(
        default_factory=list, alias="attackMethodResults"
    )


class AigReport(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False, populate_by_name=True)

    model_name: str = Field(alias="modelName", min_length=1)
    base_total: int = Field(default=0, alias="baseTotal", ge=0)
    total: int = Field(ge=0)
    jailbreak: int = Field(ge=0)
    score: float | None = Field(default=None, ge=0, le=100)
    errored: int = Field(default=0, ge=0)
    useless: int = Field(default=0, ge=0)
    extra_body: AigExtraBody = Field(default_factory=AigExtraBody, alias="extraBody")

    @model_validator(mode="after")
    def _counts_fit(self) -> AigReport:
        if self.jailbreak > self.total:
            raise ValueError("jailbreak exceeds total")
        return self

    @property
    def asr(self) -> float | None:
        return self.jailbreak / self.total if self.total else None


def parse_result(data: dict[str, Any]) -> list[AigReport]:
    payload = data.get("result", data)
    if not isinstance(payload, dict) or payload.get("msgType") != "json":
        raise AigResultError("AIG result is not a JSON payload")
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise AigResultError("AIG result has no model reports")
    try:
        return [AigReport.model_validate(item) for item in content]
    except ValidationError as exc:
        raise AigResultError(f"AIG result shape is invalid: {exc.errors(include_input=False)}") from None
