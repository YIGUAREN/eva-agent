"""Paths and files that make a run inspectable on disk."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.result import Approval
from eva_agent.domain.run import validate_run_id


@dataclass(frozen=True)
class RunPaths:
    root: Path
    evidence: Path
    plan_draft: Path
    plan_approved: Path
    target_probe: Path
    aig_dir: Path
    report_markdown: Path
    report_json: Path


def run_paths(data_dir: str | Path, run_id: str) -> RunPaths:
    validate_run_id(run_id)
    root = Path(data_dir) / "runs" / run_id
    return RunPaths(
        root=root,
        evidence=root / "evidence",
        plan_draft=root / "plan.draft.yaml",
        plan_approved=root / "plan.approved.yaml",
        target_probe=root / "target_probe.jsonl",
        aig_dir=root / "aig",
        report_markdown=root / "report.md",
        report_json=root / "report.json",
    )


_SAFE_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9._-]{1,255}")
_AIG_ARTIFACT_NAME = re.compile(r"aig/([A-Za-z0-9_-]{1,128})\.(json|csv)")


def artifact_file(data_dir: str | Path, run_id: str, name: str) -> Path:
    """Resolve a stored artifact name to a file, refusing anything outside the run."""
    paths = run_paths(data_dir, run_id)
    if _SAFE_ARTIFACT_NAME.fullmatch(name) is not None:
        base, relative = paths.root, name
    elif (match := _AIG_ARTIFACT_NAME.fullmatch(name)) is not None:
        base, relative = paths.aig_dir, f"{match.group(1)}.{match.group(2)}"
    else:
        raise ValueError(f"unsafe artifact name {name!r}")
    root = base.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"unsafe artifact name {name!r}")
    return path


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
