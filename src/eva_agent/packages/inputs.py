"""Collect the exact inputs a run used so a package can be replayed offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eva_agent.config.settings import AppConfig
from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import DatasetRef, InputKind, RunInput
from eva_agent.domain.target import TargetSpec
from eva_agent.graph.experts.rows import prepare_rows_with_indexes
from eva_agent.packages.manifest import PackageError, sha256_bytes
from eva_agent.pricing import ModelPrice, PricingError, PricingTable
from eva_agent.storage.models import RunRow
from eva_agent.tools.dataset import build_sample_dataset_tool
from eva_agent.tools.declarative import build_run_declarative_checks_tool
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.target import build_read_target_probe_tool

_EVIDENCE_MANIFEST = "manifest.json"
_PROBE_FILE = "target_probe.jsonl"


class EvidenceSnapshot(BaseModel):
    """An evidence bundle without its absolute workspace root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: InputKind
    files: list[EvidenceFile] = Field(default_factory=list)
    pasted: dict[str, str] | None = None
    dataset: DatasetRef | None = None
    target: TargetSpec | None = None
    target_prompts: list[str] = Field(default_factory=list)
    manifest_sha256: str = Field(min_length=64, max_length=64)

    def as_bundle(self) -> EvidenceBundle:
        return EvidenceBundle(
            kind=self.kind,
            root=None,
            files=list(self.files),
            pasted=self.pasted,
            dataset=self.dataset,
            target=self.target,
            target_prompts=list(self.target_prompts),
            manifest_sha256=self.manifest_sha256,
        )


class PackageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_input: RunInput
    evidence: EvidenceSnapshot
    rows: list[dict[str, str]] = Field(default_factory=list)
    source_indexes: list[int | None] = Field(default_factory=list)
    probe_records: list[ProbeRecord] = Field(default_factory=list)
    prices: list[ModelPrice] = Field(default_factory=list)


def build_package_tools(config: AppConfig) -> ToolRegistry:
    """The read-only tool subset that reproduces the grading rows."""
    registry = ToolRegistry()
    registry.register(build_read_evidence_tool(config.pipeline.dataset_max_bytes))
    registry.register(
        build_sample_dataset_tool(
            max_bytes=config.pipeline.dataset_max_bytes,
            max_rows=config.pipeline.dataset_max_rows,
        )
    )
    registry.register(build_read_target_probe_tool())
    registry.register(build_run_declarative_checks_tool(max_rows=config.pipeline.dataset_max_rows))
    return registry


def _error_detail(exc: Exception) -> str:
    """Render an error without echoing the values that failed validation."""
    if isinstance(exc, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            if error["loc"]
            else error["msg"]
            for error in exc.errors(include_input=False, include_url=False)
        )
        return details or "validation failed"
    return str(exc) or type(exc).__name__


def load_evidence_snapshot(evidence_dir: Path) -> EvidenceSnapshot:
    path = evidence_dir / _EVIDENCE_MANIFEST
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise PackageError(f"run has no readable evidence manifest: {_error_detail(exc)}") from None
    if not isinstance(payload, dict):
        raise PackageError("evidence manifest is not an object")
    try:
        return EvidenceSnapshot(
            kind=InputKind(payload.get("kind")),
            files=[EvidenceFile.model_validate(item) for item in payload.get("files") or []],
            pasted=payload.get("pasted"),
            dataset=DatasetRef.model_validate(payload["dataset"]) if payload.get("dataset") else None,
            target=TargetSpec.model_validate(payload["target"]) if payload.get("target") else None,
            target_prompts=list(payload.get("target_prompts") or []),
            manifest_sha256=sha256_bytes(raw),
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise PackageError(f"invalid evidence manifest: {_error_detail(exc)}") from None


def load_probe_records(run_root: Path) -> list[ProbeRecord]:
    path = run_root / _PROBE_FILE
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [ProbeRecord.model_validate(json.loads(line)) for line in lines if line.strip()]
    except (OSError, ValueError) as exc:
        raise PackageError(f"cannot read target probes: {_error_detail(exc)}") from None


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sample_limit(plan_json: dict[str, Any] | None, config: AppConfig) -> int:
    """The row limit the approved plan authorised, falling back to the pipeline maximum."""
    candidates: list[int] = []
    if isinstance(plan_json, dict):
        if _positive_int(plan_json.get("sample_size")):
            candidates.append(plan_json["sample_size"])
        dimensions = plan_json.get("dimensions")
        if isinstance(dimensions, list):
            for item in dimensions:
                if not isinstance(item, dict) or item.get("enabled") is False:
                    continue
                if _positive_int(item.get("sample_size")):
                    candidates.append(item["sample_size"])
    return max(candidates) if candidates else config.pipeline.dataset_max_rows


def load_prices(config: AppConfig, *, target: TargetSpec | None) -> list[ModelPrice]:
    """Package the price rows a target run needs; other inputs have no target cost to replay."""
    if target is None:
        return []
    try:
        table = PricingTable.load(Path(config.pricing.path))
    except (PricingError, OSError) as exc:
        raise PackageError(f"cannot load pricing: {_error_detail(exc)}") from None
    keys = table.keys()
    return [table.get(provider, model) for provider, model in keys]


async def collect_package_input(*, config: AppConfig, run_row: RunRow) -> PackageInput:
    try:
        run_input = RunInput.model_validate(run_row.input_json)
    except ValidationError as exc:
        raise PackageError(f"run input is invalid: {_error_detail(exc)}") from None
    run_root = Path(config.storage.data_dir) / "runs" / run_row.id
    evidence_dir = run_root / "evidence"
    evidence = load_evidence_snapshot(evidence_dir)
    tools = build_package_tools(config)
    limit = _sample_limit(run_row.plan_json, config)
    try:
        bundle = evidence.as_bundle().model_copy(update={"root": str(evidence_dir)})
        rows, indexes = await prepare_rows_with_indexes(tools, bundle, run_id=run_row.id, limit=limit)
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError(f"cannot collect the run rows: {_error_detail(exc)}") from None
    return PackageInput(
        run_input=run_input,
        evidence=evidence,
        rows=rows,
        source_indexes=indexes,
        probe_records=load_probe_records(run_root),
        prices=load_prices(config, target=evidence.target),
    )
