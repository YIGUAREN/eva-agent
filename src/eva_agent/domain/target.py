"""Target definitions for online API evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TargetKind(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    HTTP_JSON = "http_json"


class UsagePaths(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: str | None = None
    output_tokens: str | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> UsagePaths:
        if not (self.input_tokens or self.output_tokens):
            raise ValueError("usage_paths requires at least one token path")
        return self


class TargetSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: TargetKind
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)

    base_url: str | None = None
    api_key_env: str | None = None

    url: str | None = None
    method: Literal["GET", "POST"] | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    header_env: dict[str, str] = Field(default_factory=dict)
    request_template: dict[str, Any] | None = None
    response_path: str | None = None
    usage_paths: UsagePaths | None = None

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> TargetSpec:
        if self.kind is TargetKind.OPENAI_COMPATIBLE:
            if not self.base_url:
                raise ValueError("openai_compatible target requires base_url")
            if any(
                (
                    self.url,
                    self.method,
                    self.headers,
                    self.header_env,
                    self.request_template,
                    self.response_path,
                    self.usage_paths,
                )
            ):
                raise ValueError("openai_compatible target must not set http_json fields")
        else:
            if not self.url or not self.request_template or not self.response_path:
                raise ValueError("http_json target requires url, request_template, response_path")
            if not self.method:
                raise ValueError("http_json target requires method (GET or POST)")
            if self.base_url or self.api_key_env:
                raise ValueError("http_json target must not set openai_compatible fields")
            if not _contains_prompt(self.request_template):
                raise ValueError("request_template must contain a {{prompt}} placeholder")
            if self.method == "GET":
                for value in self.request_template.values():
                    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                        raise ValueError("GET request_template values must be strings or numbers")

        if set(self.headers) & set(self.header_env):
            raise ValueError("headers and header_env must not share a key")
        _reject_credential_headers(self.headers)
        return self


_CREDENTIAL_HEADER_SEGMENTS = frozenset(
    {
        "auth",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "key",
        "password",
        "passwd",
        "secret",
        "signature",
        "token",
    }
)


def _reject_credential_headers(headers: dict[str, str]) -> None:
    for name in headers:
        segments = name.strip().lower().replace("_", "-").split("-")
        if any(segment in _CREDENTIAL_HEADER_SEGMENTS for segment in segments):
            raise ValueError(f"literal header {name!r} may carry a credential; put it in header_env instead")


def _contains_prompt(value: Any) -> bool:
    if isinstance(value, str):
        return "{{prompt}}" in value
    if isinstance(value, dict):
        return any(_contains_prompt(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_prompt(item) for item in value)
    return False
