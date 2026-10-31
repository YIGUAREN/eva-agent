"""Collect the exact inputs a run used so a package can be replayed offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eva_agent.config.settings import AppConfig
from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.plan import Dimension
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import DatasetRef, InputKind, RunInput
from eva_agent.domain.target import TargetSpec
from eva_agent.graph.experts.rows import prepare_rows_with_indexes
from eva_agent.packages.manifest import PackageError, sha256_bytes
from eva_agent.pricing import ModelPrice, PricingError, PricingTable
from eva_agent.runtime.dependencies import QUALITY_DEFAULT_SAMPLE_SIZE
from eva_agent.storage.models import RunRow
from eva_agent.tools.dataset import build_sample_dataset_tool
from eva_agent.tools.declarative import build_run_declarative_checks_tool
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.target import build_read_target_probe_tool

_EVIDENCE_MANIFEST = "manifest.json"
_PROBE_FILE = "target_probe.jsonl"
# ``TargetSpec`` requires a request template for HTTP targets and a template
# containing a prompt placeholder. Replay never reads the body, so a fixed
# placeholder keeps the model valid without carrying the original values.
_PLACEHOLDER_REQUEST_TEMPLATE = {"prompt": "{{prompt}}"}


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
    # The effective row limit each enabled expert used, keyed by Dimension value.
    dimension_limits: dict[str, int] = Field(default_factory=dict)


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
    """Render an error without echoing validation inputs or filesystem paths."""
    if isinstance(exc, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            if error["loc"]
            else error["msg"]
            for error in exc.errors(include_input=False, include_url=False)
        )
        return details or "validation failed"
    if isinstance(exc, OSError):
        # ``str(OSError)`` embeds the absolute filename; PackageError text can
        # reach package manifests, which must never carry one.
        return exc.strerror or type(exc).__name__
    return str(exc) or type(exc).__name__


def _without_target_secrets(target: TargetSpec | None) -> TargetSpec | None:
    """Strip every target field that can carry a credential from the snapshot.

    Replay only needs ``kind``/``provider``/``model`` for pricing, so header
    values, header env references, the request body and the URL query/fragment
    (where API keys are commonly passed) never enter the package. The scheme,
    host and path stay because the public report and evidence display them.
    """
    if target is None:
        return None
    updates: dict[str, Any] = {"headers": {}, "header_env": {}}
    if target.url is not None:
        parts = urlsplit(target.url)
        updates["url"] = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    if target.request_template is not None:
        updates["request_template"] = dict(_PLACEHOLDER_REQUEST_TEMPLATE)
    return target.model_copy(update=updates)


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
            target=_without_target_secrets(
                TargetSpec.model_validate(payload["target"]) if payload.get("target") else None
            ),
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


def _enabled_dimension_plans(plan_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(plan_json, dict):
        return []
    dimensions = plan_json.get("dimensions")
    if not isinstance(dimensions, list):
        return []
    return [
        item for item in dimensions if isinstance(item, dict) and item.get("enabled") is not False
    ]


def _dimension_limits(
    plan_json: dict[str, Any] | None,
    *,
    probe_count: int,
) -> dict[str, int]:
    """The effective row limit each expert used, keyed by Dimension value.

    Quality and security fall back to ``GraphDependencies.default_sample_size``;
    latency and cost read every probe record unless the plan declares a limit.
    """
    limits: dict[str, int] = {}
    for item in _enabled_dimension_plans(plan_json):
        try:
            dimension = Dimension(item.get("dimension"))
        except ValueError:
            raise PackageError("plan declares an unknown dimension") from None
        declared = item.get("sample_size")
        if isinstance(declared, int) and not isinstance(declared, bool) and declared > 0:
            limit = declared
        elif dimension in (Dimension.QUALITY, Dimension.SECURITY):
            limit = QUALITY_DEFAULT_SAMPLE_SIZE
        else:
            limit = probe_count
        limits[dimension.value] = limit
    return limits


def _packaged_run_input(run_input: RunInput, evidence: EvidenceSnapshot) -> RunInput:
    """Keep only the evidence-relative dataset name; the raw CLI path is absolute."""
    if run_input.kind is not InputKind.DATASET or run_input.dataset is None:
        return run_input
    if evidence.dataset is None:
        raise PackageError("dataset evidence has no dataset reference")
    return run_input.model_copy(update={"dataset": evidence.dataset})


def load_prices(config: AppConfig, *, target: TargetSpec | None) -> list[ModelPrice]:
    """Package the one price row a target run used; other inputs have no target cost to replay."""
    if target is None:
        return []
    try:
        table = PricingTable.load(Path(config.pricing.path))
        return [table.get(target.provider, target.model)]
    except (PricingError, OSError) as exc:
        raise PackageError(f"cannot load pricing: {_error_detail(exc)}") from None


async def collect_package_input(*, config: AppConfig, run_row: RunRow) -> PackageInput:
    try:
        run_input = RunInput.model_validate(run_row.input_json)
    except ValidationError as exc:
        raise PackageError(f"run input is invalid: {_error_detail(exc)}") from None
    run_root = Path(config.storage.data_dir) / "runs" / run_row.id
    evidence_dir = run_root / "evidence"
    evidence = load_evidence_snapshot(evidence_dir)
    tools = build_package_tools(config)
    probe_records = load_probe_records(run_root)
    dimension_limits = _dimension_limits(run_row.plan_json, probe_count=len(probe_records))
    limit = max(dimension_limits.values(), default=config.pipeline.dataset_max_rows)
    try:
        bundle = evidence.as_bundle().model_copy(update={"root": str(evidence_dir)})
        rows, indexes = await prepare_rows_with_indexes(tools, bundle, run_id=run_row.id, limit=limit)
    except PackageError:
        raise
    except Exception as exc:
        raise PackageError(f"cannot collect the run rows: {_error_detail(exc)}") from None
    return PackageInput(
        run_input=_packaged_run_input(run_input, evidence),
        evidence=evidence,
        rows=rows,
        source_indexes=indexes,
        probe_records=probe_records,
        prices=load_prices(config, target=evidence.target),
        dimension_limits=dimension_limits,
    )
