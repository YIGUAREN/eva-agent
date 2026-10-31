"""Build the AIG create-task payload without exposing secrets in errors."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import yaml

from eva_agent.config.settings import (
    AigAgentScanSettings,
    AigScanSettings,
    AigSkillScanSettings,
    ModelConfig,
)
from eva_agent.domain.target import TargetSpec

_TRANSFORM_PATH = re.compile(r"[A-Za-z0-9_]+(?:\[\d+\])?(?:\.[A-Za-z0-9_]+(?:\[\d+\])?)*")


def build_create_body(
    *,
    target: TargetSpec,
    scan_base_url: str,
    eval_model: ModelConfig,
    eval_token: str,
    target_token: str,
    scan: AigScanSettings,
) -> dict[str, Any]:
    return {
        "type": "model_redteam_report",
        "content": {
            "model": [
                {
                    "model": target.model,
                    "token": target_token,
                    "base_url": scan_base_url,
                }
            ],
            "eval_model": {
                "model": eval_model.model,
                "token": eval_token,
                "base_url": eval_model.base_url or "",
            },
            "dataset": {
                "dataFile": list(scan.datasets),
                "numPrompts": scan.num_prompts,
                "randomSeed": scan.random_seed,
            },
            "techniques": [],
        },
    }


def response_transform(response_path: str) -> str:
    """Turn a ``$.a.b[0].c`` response path into AIG's ``a.b[0].c`` transform path."""
    path = response_path.strip()
    if path.startswith("$."):
        path = path[2:]
    if _TRANSFORM_PATH.fullmatch(path) is None:
        raise ValueError("response_path is not a simple dotted path")
    return path


def _split_target_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("target url must be an absolute http(s) url")
    endpoint = parsed.path
    if parsed.query:
        endpoint = f"{endpoint}?{parsed.query}"
    return f"{parsed.scheme}://{parsed.netloc}", endpoint


def build_agent_config(*, target: TargetSpec, headers: Mapping[str, str]) -> str:
    """Render the ``targets`` YAML accepted by AIG's agent adapter."""
    if target.url is None or target.response_path is None:
        raise ValueError("agent_http target requires url and response_path")
    url, endpoint = _split_target_url(target.url)
    config: dict[str, Any] = {
        "url": url,
        "method": target.method or "POST",
        "headers": {"Content-Type": "application/json", **headers},
        "body": target.request_template or {},
        "transform_response": response_transform(target.response_path),
    }
    if endpoint:
        config["endpoint"] = endpoint
    return yaml.safe_dump({"targets": [{"id": "http", "config": config}]}, allow_unicode=True, sort_keys=False)


def _eval_model_block(eval_model: ModelConfig, eval_token: str) -> dict[str, str]:
    return {"model": eval_model.model, "token": eval_token, "base_url": eval_model.base_url or ""}


def build_agent_scan_body(
    *,
    target: TargetSpec,
    headers: Mapping[str, str],
    eval_model: ModelConfig,
    eval_token: str,
    scan: AigAgentScanSettings,
) -> dict[str, Any]:
    return {
        "type": "agent_scan",
        "content": {
            "agent_config": build_agent_config(target=target, headers=headers),
            "eval_model": _eval_model_block(eval_model, eval_token),
            "language": scan.language,
            "prompt": scan.prompt,
            "skills": list(scan.skills),
        },
    }


def build_skill_scan_body(
    *,
    eval_model: ModelConfig,
    eval_token: str,
    file_url: str,
    scan: AigSkillScanSettings,
) -> dict[str, Any]:
    return {
        "type": "skill_scan",
        "content": {
            "model": _eval_model_block(eval_model, eval_token),
            "language": scan.language,
            "prompt": scan.prompt,
            "attachments": file_url,
        },
    }
