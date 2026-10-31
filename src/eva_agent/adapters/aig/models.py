"""Validated AIG model-redteam result models."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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

    vulnerability_results: list[AigVulnerability] = Field(default_factory=list, alias="vulnerabilityResults")
    attack_method_results: list[AigAttackMethod] = Field(default_factory=list, alias="attackMethodResults")


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


class AigSkillFinding(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)

    title: str = Field(min_length=1, max_length=2_048)
    description: str = Field(min_length=1, max_length=8_192)
    risk_type: str = Field(min_length=1, max_length=512)
    level: str = Field(default="", max_length=64)
    suggestion: str = Field(default="", max_length=8_192)
    file: str | None = Field(default=None, max_length=1_024)
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)

    @field_validator("file")
    @classmethod
    def _relative_file(cls, value: str | None) -> str | None:
        if value is None:
            return value
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("finding file must stay inside the package")
        return value


class AigSkillReport(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)

    score: int = Field(ge=0, le=100)
    language: str = Field(default="", max_length=64)
    llm: str = Field(default="", max_length=128)
    readme: str = Field(default="", max_length=32_768)
    results: list[AigSkillFinding] = Field(default_factory=list)


class AigAgentFinding(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)

    id: str = Field(default="", max_length=64)
    type: str = Field(default="", max_length=512)
    title: str = Field(min_length=1, max_length=2_048)
    description: str = Field(min_length=1, max_length=8_192)
    level: Literal["HIGH", "MEDIUM", "LOW"]
    owasp: list[str] = Field(default_factory=list)
    suggestion: str = Field(default="", max_length=8_192)


class AigAgentReport(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)

    schema_version: str = Field(min_length=1, max_length=128)
    agent_name: str = Field(default="", max_length=256)
    score: int = Field(ge=0, le=100)
    risk_type: str = Field(default="safe", max_length=32)
    total_tests: int = Field(default=0, ge=0)
    vulnerable_tests: int = Field(default=0, ge=0)
    results: list[AigAgentFinding] = Field(default_factory=list)
    owasp_agentic_2026_top10: list[dict[str, Any]] = Field(default_factory=list)
    report_description: str = Field(default="", max_length=32_768)
    language: str = Field(default="", max_length=64)


def _direct_result(data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("result")
    if payload is None and isinstance(data.get("data"), dict):
        # Accept a raw HTTP envelope as well as the client's unwrapped ``data``.
        payload = data["data"].get("result")
    if not isinstance(payload, dict):
        raise AigResultError("AIG result has no structured result object")
    return payload


def _validated(model: type[BaseModel], payload: dict[str, Any]) -> Any:
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise AigResultError(f"AIG result shape is invalid: {exc.errors(include_input=False)}") from None


def parse_skill_result(data: dict[str, Any]) -> AigSkillReport:
    return _validated(AigSkillReport, _direct_result(data))


def parse_agent_result(data: dict[str, Any]) -> AigAgentReport:
    return _validated(AigAgentReport, _direct_result(data))
