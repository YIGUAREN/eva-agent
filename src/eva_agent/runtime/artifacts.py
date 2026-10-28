"""Paths and files that make a run inspectable on disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.result import Approval


@dataclass(frozen=True)
class RunPaths:
    root: Path
    evidence: Path
    plan_draft: Path
    plan_approved: Path
    report_markdown: Path
    report_json: Path


def run_paths(data_dir: str | Path, run_id: str) -> RunPaths:
    root = Path(data_dir) / "runs" / run_id
    return RunPaths(
        root=root,
        evidence=root / "evidence",
        plan_draft=root / "plan.draft.yaml",
        plan_approved=root / "plan.approved.yaml",
        report_markdown=root / "report.md",
        report_json=root / "report.json",
    )


def write_plan_draft(paths: RunPaths, plan: EvalPlan, *, overwrite: bool = False) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.plan_draft.exists() and not overwrite:
        return
    payload = plan.model_dump(mode="json")
    paths.plan_draft.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def read_plan_draft(paths: RunPaths) -> EvalPlan:
    raw = yaml.safe_load(paths.plan_draft.read_text(encoding="utf-8"))
    return EvalPlan.model_validate(raw)


def write_plan_approved(paths: RunPaths, approval: Approval) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    payload = approval.model_dump(mode="json")
    paths.plan_approved.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
