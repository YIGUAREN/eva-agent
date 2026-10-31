"""Build, verify, and read the grader package described in spec §14.2."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from eva_agent.config.settings import AppConfig
from eva_agent.domain.evidence import SKILL_MANIFEST_NAMES
from eva_agent.domain.plan import EvalPlan
from eva_agent.packages.inputs import EvidenceSnapshot, PackageInput, collect_package_input
from eva_agent.packages.manifest import (
    FileEntry,
    PackageError,
    PackageManifest,
    canonical_json,
    extract_zip,
    sha256_file,
    verify_tree,
    write_manifest,
    write_zip,
)
from eva_agent.packages.snapshots import SnapshotSet, collect_snapshots
from eva_agent.storage.models import RunRow, utcnow

README = """# Grader package

This package records one evaluation run so it can be replayed offline.

- `plan.json` — the approved plan.
- `input.json` — the run input, the exact grading rows, the probe records and the price rows.
- `snapshots/gradings.json` — recorded grader outcomes; replay never calls a model.
- `report.json` — the report produced by the run.

Run `eva replay <package.zip> --out <dir>` to verify the package.

Replay compares only fields it can re-derive: quality `score`, `confidence` and
`metrics`; security `metrics.passive` (plus `score` and `confidence` when no
score cap applies and no active scan ran); and cost and latency `status`,
`score`, `confidence`, `summary`, `metrics`, `findings`, `error` and
`truncated_reason`. It does not compare `usage`, `evidence` locators,
`generated_at`, the report-level `overall_score`, `probe`, or
`evidence_summary`.
"""

_MEMBER_NAMES = ("README.md", "input.json", "plan.json", "report.json", "snapshots/gradings.json")


@dataclass(frozen=True)
class GraderPackage:
    manifest: PackageManifest
    plan: EvalPlan
    inputs: PackageInput
    snapshots: SnapshotSet
    report: dict


def _plan_from_row(run_row: RunRow) -> EvalPlan:
    if not run_row.plan_json:
        raise PackageError("run has no approved plan")
    return EvalPlan.model_validate(run_row.plan_json)


def _replay_blockers(snapshot_blockers: list[str], report: dict, evidence: EvidenceSnapshot) -> list[str]:
    """Every reason this package cannot be replayed offline and compared exactly."""
    blockers = list(snapshot_blockers)
    for dimension in report.get("truncated_dimensions") or []:
        blockers.append(f"{dimension}: the dimension was truncated by the budget")
    for item in report.get("dimensions") or []:
        if item.get("dimension") != "security":
            continue
        active = (item.get("metrics") or {}).get("active") or {}
        if active.get("status") not in (None, "not_applicable"):
            blockers.append("security: replay does not cover an active AIG scan")
    if any(PurePosixPath(file.path).name in SKILL_MANIFEST_NAMES for file in evidence.files):
        blockers.append("security: replay does not cover skill threat rows")
    return blockers


async def build_grader_package(
    *, config: AppConfig, run_row: RunRow, out_dir: Path
) -> tuple[Path, PackageManifest]:
    plan = _plan_from_row(run_row)
    inputs = await collect_package_input(config=config, run_row=run_row)
    snapshots, snapshot_blockers = collect_snapshots(
        data_dir=config.storage.data_dir,
        run_id=run_row.id,
        dimensions=[item for item in plan.dimensions if item.enabled],
        rows=inputs.rows,
        dimension_limits=inputs.dimension_limits,
    )
    report_path = Path(config.storage.data_dir) / "runs" / run_row.id / "report.json"
    if not report_path.is_file():
        raise PackageError("run has no report.json; only completed runs can be packaged")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    blockers = _replay_blockers(snapshot_blockers, report, inputs.evidence)
    workdir = Path(tempfile.mkdtemp(prefix="eva-package-"))
    try:
        (workdir / "snapshots").mkdir(parents=True)
        (workdir / "plan.json").write_bytes(canonical_json(plan.model_dump(mode="json")))
        (workdir / "input.json").write_bytes(canonical_json(inputs.model_dump(mode="json")))
        (workdir / "snapshots" / "gradings.json").write_bytes(
            canonical_json(snapshots.model_dump(mode="json"))
        )
        (workdir / "report.json").write_bytes(canonical_json(report))
        (workdir / "README.md").write_text(README, encoding="utf-8")
        files = {
            name: FileEntry(sha256=sha256_file(workdir / name), size=(workdir / name).stat().st_size)
            for name in _MEMBER_NAMES
        }
        catalog = Path(config.openjudge.catalog_path)
        manifest = PackageManifest(
            kind="grader_package",
            run_id=run_row.id,
            created_at=utcnow().isoformat(),
            report_version=int(report.get("version") or 0),
            catalog_sha256=sha256_file(catalog) if catalog.is_file() else None,
            evidence_sha256=inputs.evidence.manifest_sha256,
            replayable=not blockers,
            replay_blockers=blockers,
            files=files,
        )
        write_manifest(workdir, manifest)
        verify_tree(workdir, manifest)
        out_dir.mkdir(parents=True, exist_ok=True)
        archive = write_zip(workdir, out_dir / f"{run_row.id}.grader-package.zip", manifest=manifest)
        return archive, manifest
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def read_grader_package(archive: Path) -> GraderPackage:
    with tempfile.TemporaryDirectory(prefix="eva-replay-") as temporary:
        workdir = Path(temporary) / "package"
        manifest = extract_zip(archive, workdir)
        plan = EvalPlan.model_validate_json((workdir / "plan.json").read_text(encoding="utf-8"))
        inputs = PackageInput.model_validate_json((workdir / "input.json").read_text(encoding="utf-8"))
        snapshots = SnapshotSet.model_validate_json(
            (workdir / "snapshots" / "gradings.json").read_text(encoding="utf-8")
        )
        report = json.loads((workdir / "report.json").read_text(encoding="utf-8"))
    return GraderPackage(manifest=manifest, plan=plan, inputs=inputs, snapshots=snapshots, report=report)
