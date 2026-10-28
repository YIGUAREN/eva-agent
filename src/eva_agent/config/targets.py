"""Load named target definitions from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from eva_agent.config.settings import ConfigError
from eva_agent.domain.target import TargetKind, TargetSpec


def load_targets(path: Path) -> dict[str, TargetSpec]:
    if not path.is_file():
        raise ConfigError(f"targets file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    entries = raw.get("targets") if isinstance(raw, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ConfigError(f"targets file has no targets: {path}")

    targets: dict[str, TargetSpec] = {}
    for name, payload in entries.items():
        if not isinstance(name, str) or not name:
            raise ConfigError(f"target name must be a non-empty string in {path}")
        if not isinstance(payload, dict):
            raise ConfigError(f"target '{name}' must be a mapping")
        if "name" in payload:
            raise ConfigError(f"target '{name}' must not define 'name'")
        data: dict[str, Any] = {"name": name, **payload}
        if data.get("kind") == TargetKind.HTTP_JSON.value:
            data.setdefault("method", "POST")
        try:
            targets[name] = TargetSpec.model_validate(data)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_input=False, include_url=False)
            )
            raise ConfigError(f"invalid target '{name}' in {path}: {details}") from None
    return targets
