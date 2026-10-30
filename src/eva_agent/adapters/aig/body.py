"""Build the AIG create-task payload without exposing secrets in errors."""

from __future__ import annotations

from typing import Any

from eva_agent.config.settings import AigScanSettings, ModelConfig
from eva_agent.domain.target import TargetSpec


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
