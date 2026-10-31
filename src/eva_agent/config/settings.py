"""Typed application configuration loaded from YAML."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


class MissingSecretError(ValueError):
    """Raised when a referenced environment variable is not set."""


class ModelConfig(BaseModel):
    """One model endpoint: agent, judge, or attack."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # "litellm" means a LiteLLM proxy that exposes an OpenAI-compatible API.
    provider: Literal["openai-compatible", "anthropic", "litellm"]
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str = Field(min_length=1)

    @model_validator(mode="after")
    def _openai_compatible_requires_base_url(self) -> ModelConfig:
        if self.provider in ("openai-compatible", "litellm") and not self.base_url:
            raise ValueError(f"provider '{self.provider}' requires base_url")
        return self


class ModelsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: ModelConfig
    judge: ModelConfig
    attack: ModelConfig


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(gt=0)


class StorageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    data_dir: str = Field(default="./data", min_length=1)
    checkpoint_path: str = Field(default="./data/checkpoints.sqlite", min_length=1)


class PipelineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_max_bytes: int = Field(default=52_428_800, gt=0)
    dataset_max_rows: int = Field(default=10_000, gt=0)
    evidence_sample_rows: int = Field(default=5, gt=0)


class OpenJudgeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_path: str = Field(default="config/graders.yaml", min_length=1)
    max_concurrency: int = Field(default=8, gt=0)
    grader_timeout_s: float = Field(default=120.0, gt=0)


class TargetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    targets_path: str = Field(default="config/targets.yaml", min_length=1)
    timeout_s: float = Field(default=30.0, gt=0)
    max_response_bytes: int = Field(default=1_048_576, gt=0)
    token_encoding: str = Field(default="cl100k_base", min_length=1)


class PricingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(default="config/pricing.yaml", min_length=1)


class SecuritySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_path: str = Field(default="config/security_policy.yaml", min_length=1)


class AigScanSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: list[str] = Field(
        default_factory=lambda: ["JailbreakPrompts-Tiny"],
        min_length=1,
    )
    num_prompts: int = Field(default=20, gt=0)
    random_seed: int = 42


class AigSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = Field(default="http://127.0.0.1:8088", min_length=1)
    timeout_s: float = Field(default=30.0, gt=0)
    poll_initial_s: float = Field(default=2.0, gt=0)
    poll_max_s: float = Field(default=10.0, gt=0)
    max_wait_s: float = Field(default=600.0, gt=0)
    min_wait_s: float = Field(default=30.0, gt=0)
    scan: AigScanSettings = Field(default_factory=AigScanSettings)

    @model_validator(mode="after")
    def _validate_windows(self) -> AigSettings:
        if self.poll_max_s < self.poll_initial_s:
            raise ValueError("poll_max_s must be >= poll_initial_s")
        if self.min_wait_s > self.max_wait_s:
            raise ValueError("min_wait_s must be <= max_wait_s")
        return self


class UploadsSettings(BaseModel):
    """Extraction limits enforced by the sandbox runner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_extracted_bytes: int = Field(default=536_870_912, gt=0)
    max_files: int = Field(default=10_000, gt=0)
    max_depth: int = Field(default=20, gt=0)


class WebSettings(BaseModel):
    """HTTP server and worker-loop settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_ttl_s: int = Field(default=43_200, gt=0)
    session_absolute_ttl_s: int = Field(default=604_800, gt=0)
    cookie_secure: bool = True
    max_concurrent_runs: int = Field(default=2, gt=0)
    poll_interval_s: float = Field(default=0.5, gt=0)
    upload_max_bytes: int = Field(default=52_428_800, gt=0)
    upload_timeout_s: float = Field(default=120.0, gt=0)
    sandbox_runner_url: str = Field(default="http://127.0.0.1:8090", min_length=1)
    sandbox_timeout_s: float = Field(default=120.0, gt=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    public_base_url: str = Field(min_length=1)
    retention_days: int = Field(gt=0)
    models: ModelsSettings
    agent: AgentSettings
    storage: StorageSettings
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    openjudge: OpenJudgeSettings = Field(default_factory=OpenJudgeSettings)
    target: TargetSettings = Field(default_factory=TargetSettings)
    pricing: PricingSettings = Field(default_factory=PricingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    web: WebSettings = Field(default_factory=WebSettings)
    uploads: UploadsSettings = Field(default_factory=UploadsSettings)
    aig: AigSettings | None = None


def load_config(path: Path) -> AppConfig:
    """Load and validate the YAML config at *path*."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {path}")

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        # Do not expose input values: a stray secret in the YAML would otherwise
        # reach logs through Pydantic's error rendering.
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" if error["loc"] else error["msg"]
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise ConfigError(f"invalid config in {path}: {details}") from None


def resolve_api_key(model: ModelConfig, env: Mapping[str, str] | None = None) -> str:
    """Return the API key named by *model.api_key_env*."""
    source = os.environ if env is None else env
    value = source.get(model.api_key_env)
    if not value:
        raise MissingSecretError(f"environment variable {model.api_key_env} is not set")
    return value


def require_sandbox_token(env: Mapping[str, str] | None = None) -> str:
    """Return SANDBOX_TOKEN from the environment; never accept it from YAML."""
    source = os.environ if env is None else env
    value = source.get("SANDBOX_TOKEN", "")
    if not value:
        raise ConfigError("environment variable SANDBOX_TOKEN is not set")
    return value


def validate_sandbox_url(value: str) -> str:
    """Return *value* when it is a usable http(s) base URL."""
    try:
        parts = urlsplit(value)
    except ValueError:
        raise ConfigError("web.sandbox_runner_url must be a valid http(s) URL") from None
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ConfigError("web.sandbox_runner_url must be a valid http(s) URL")
    return value
