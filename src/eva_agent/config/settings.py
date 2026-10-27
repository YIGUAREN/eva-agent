"""Typed application configuration loaded from YAML."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


class MissingSecretError(ValueError):
    """Raised when a referenced environment variable is not set."""


class ModelConfig(BaseModel):
    """One model endpoint: agent, judge, or attack."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai-compatible", "anthropic", "litellm"]
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str = Field(min_length=1)

    @model_validator(mode="after")
    def _litellm_requires_base_url(self) -> ModelConfig:
        if self.provider == "litellm" and not self.base_url:
            raise ValueError("provider 'litellm' requires base_url")
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


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    public_base_url: str = Field(min_length=1)
    retention_days: int = Field(gt=0)
    models: ModelsSettings
    agent: AgentSettings
    storage: StorageSettings


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
        raise ConfigError(str(exc)) from exc


def resolve_api_key(model: ModelConfig, env: Mapping[str, str] | None = None) -> str:
    """Return the API key named by *model.api_key_env*."""
    source = os.environ if env is None else env
    value = source.get(model.api_key_env)
    if not value:
        raise MissingSecretError(f"environment variable {model.api_key_env} is not set")
    return value
