"""Resolve target headers from environment variables without persisting values."""

from __future__ import annotations

from collections.abc import Mapping

from eva_agent.domain.target import TargetKind, TargetSpec


class MissingSecretError(ValueError):
    """Raised when a required environment variable is missing."""


def validate_target_env(target: TargetSpec, env: Mapping[str, str]) -> None:
    if target.api_key_env and not env.get(target.api_key_env):
        raise MissingSecretError(f"environment variable {target.api_key_env} is not set")
    for name in target.header_env.values():
        if not env.get(name):
            raise MissingSecretError(f"environment variable {name} is not set")


def resolve_headers(target: TargetSpec, env: Mapping[str, str]) -> dict[str, str]:
    if target.kind is TargetKind.OPENAI_COMPATIBLE:
        if not target.api_key_env:
            return {}
        key = env.get(target.api_key_env)
        if not key:
            raise MissingSecretError(f"environment variable {target.api_key_env} is not set")
        return {"Authorization": f"Bearer {key}"}

    headers = dict(target.headers)
    for header, name in target.header_env.items():
        value = env.get(name)
        if not value:
            raise MissingSecretError(f"environment variable {name} is not set")
        headers[header] = value
    return headers
