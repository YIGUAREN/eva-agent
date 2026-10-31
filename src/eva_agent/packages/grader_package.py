"""Build, verify, and read the grader package described in spec §14.2."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import uuid4

from eva_agent.config.settings import AppConfig
from eva_agent.domain.evidence import SKILL_MANIFEST_NAMES
from eva_agent.domain.plan import Dimension, EvalPlan
from eva_agent.packages.inputs import (
    EvidenceSnapshot,
    PackageInput,
    _error_detail,
    collect_package_input,
)
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
from eva_agent.runtime.artifacts import run_paths
from eva_agent.storage.models import RunRow, utcnow

README = """# 评测包

本包记录一次评测运行，用于离线复现校验。

- `plan.json` —— 本次运行批准的评测计划。
- `input.json` —— 运行输入、实际参与评分的样本行、探测（probe）记录与价格行。
- `snapshots/gradings.json` —— 已记录的评分器输出；复现过程不会调用任何模型。
- `report.json` —— 本次运行生成的报告。

执行 `eva replay <包.zip> --out <目录>` 校验本包。

复现只比较可重新推导的字段：质量维度的 `score`、`confidence` 与 `metrics`；安全维度的
`metrics.passive`（未触发分数上限且未执行主动扫描时，另比较 `score` 与 `confidence`）；
成本与延迟维度的 `status`、`score`、`confidence`、`summary`、`metrics`、`findings`、
`error` 与 `truncated_reason`。以下字段不参与比较：`usage`、证据定位信息（locator）、
`generated_at`、报告级 `overall_score`、`probe`、`evidence_summary`。
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


def _dimension_input_count(inputs: PackageInput, dimension: str, limit: int) -> int:
    """How many inputs the expert for *dimension* graded.

    Quality and security slice ``rows``; cost and latency slice the probe
    records. An empty slice means the expert failed instead of summarizing.
    """
    if dimension in (Dimension.QUALITY.value, Dimension.SECURITY.value):
        return len(inputs.rows[:limit])
    return len(inputs.probe_records[:limit])


def _replay_blockers(
    snapshot_blockers: list[str], report: dict, evidence: EvidenceSnapshot, inputs: PackageInput
) -> list[str]:
    """Every reason this package cannot be replayed offline and compared exactly."""
    blockers = list(snapshot_blockers)
    for dimension, limit in inputs.dimension_limits.items():
        if _dimension_input_count(inputs, dimension, limit) == 0:
            blockers.append(f"{dimension}: the dimension has no inputs to replay")
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


def _is_absolute_path(value: str) -> bool:
    """Whether *value* is an absolute POSIX or Windows path (a URL is neither)."""
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _relative_display(value: str) -> str:
    """Replace an absolute path with its basename, preserving any ``#fragment``."""
    path, separator, fragment = value.partition("#")
    if not _is_absolute_path(path):
        return value
    name = PureWindowsPath(path).name or "file"
    return f"{name}#{fragment}" if separator else name


def _iter_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _scrub_evidence(container: dict[str, Any]) -> None:
    for finding in container.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        for reference in finding.get("evidence") or []:
            if isinstance(reference, dict) and isinstance(reference.get("locator"), str):
                reference["locator"] = _relative_display(reference["locator"])
    for reference in container.get("evidence") or []:
        if isinstance(reference, dict) and isinstance(reference.get("locator"), str):
            reference["locator"] = _relative_display(reference["locator"])


def _scrub_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove absolute filesystem paths from the fields a report can carry them in.

    Replay never compares ``probe`` or evidence locators, so a relative display
    name is enough provenance and cannot leak the run's data directory. Any
    remaining absolute string fails the build closed.
    """
    scrubbed: dict[str, Any] = copy.deepcopy(report)
    probe = scrubbed.get("probe")
    if isinstance(probe, dict) and isinstance(probe.get("path"), str):
        probe["path"] = _relative_display(probe["path"])
    for dimension in scrubbed.get("dimensions") or []:
        if not isinstance(dimension, dict):
            continue
        metrics = dimension.get("metrics")
        if isinstance(metrics, dict):
            active = metrics.get("active")
            if isinstance(active, dict) and isinstance(active.get("raw_result_path"), str):
                active["raw_result_path"] = _relative_display(active["raw_result_path"])
        _scrub_evidence(dimension)
    _scrub_evidence(scrubbed)
    if any(_is_absolute_path(text) for text in _iter_strings(scrubbed)):
        raise PackageError("report.json contains an absolute path")
    return scrubbed


def _catalog_sha256(config: AppConfig) -> str | None:
    """Hash the configured grader catalog, failing closed when it cannot be read."""
    configured = config.openjudge.catalog_path
    if not configured:
        return None
    try:
        return sha256_file(Path(configured))
    except OSError as exc:
        raise PackageError(f"cannot read the grader catalog: {_error_detail(exc)}") from None


def _report_payload(report_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"cannot read the run report: {_error_detail(exc)}") from None
    if not isinstance(raw, dict):
        raise PackageError("run report is not an object")
    return _scrub_report(raw)


async def build_grader_package(
    *, config: AppConfig, run_row: RunRow, out_dir: Path
) -> tuple[Path, PackageManifest]:
    try:
        try:
            paths = run_paths(config.storage.data_dir, run_row.id)
        except ValueError as exc:
            raise PackageError(str(exc)) from None
        plan = _plan_from_row(run_row)
        inputs = await collect_package_input(config=config, run_row=run_row)
        snapshots, snapshot_blockers = collect_snapshots(
            data_dir=config.storage.data_dir,
            run_id=run_row.id,
            dimensions=[item for item in plan.dimensions if item.enabled],
            rows=inputs.rows,
            dimension_limits=inputs.dimension_limits,
        )
        if not paths.report_json.is_file():
            raise PackageError("run has no report.json; only completed runs can be packaged")
        report = _report_payload(paths.report_json)
        blockers = _replay_blockers(snapshot_blockers, report, inputs.evidence, inputs)
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
                name: FileEntry(
                    sha256=sha256_file(workdir / name), size=(workdir / name).stat().st_size
                )
                for name in _MEMBER_NAMES
            }
            manifest = PackageManifest(
                kind="grader_package",
                run_id=run_row.id,
                created_at=utcnow().isoformat(),
                report_version=int(report.get("version") or 0),
                catalog_sha256=_catalog_sha256(config),
                evidence_sha256=inputs.evidence.manifest_sha256,
                replayable=not blockers,
                replay_blockers=blockers,
                files=files,
            )
            write_manifest(workdir, manifest)
            verify_tree(workdir, manifest)
            out_dir.mkdir(parents=True, exist_ok=True)
            destination = out_dir / f"{run_row.id}.grader-package.zip"
            # Write beside the destination and move it into place only on
            # success, so a mid-write failure cannot truncate a previous
            # archive with the same name. This mirrors ``build_run_bundle``.
            staging = out_dir / f".{destination.name}.tmp-{uuid4().hex}"
            try:
                write_zip(workdir, staging, manifest=manifest)
                os.replace(staging, destination)
            except BaseException:
                staging.unlink(missing_ok=True)
                raise
            return destination, manifest
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    except OSError as exc:
        # ``str(OSError)`` embeds the absolute filename; the OS reason is enough
        # and safe for both the CLI and the admin HTTP route.
        raise PackageError(f"cannot write grader package: {_error_detail(exc)}") from None


def _require_grader_package(manifest: PackageManifest) -> None:
    if manifest.kind != "grader_package":
        raise PackageError("package is not a grader package")
    missing = sorted(set(_MEMBER_NAMES) - set(manifest.files))
    if missing:
        raise PackageError(f"package is missing required members {missing}")


def _load_member[T](workdir: Path, name: str, parse: Callable[[str], T]) -> T:
    """Read and parse one member, translating every parse failure to ``PackageError``."""
    try:
        return parse((workdir / name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"cannot read {name}: {_error_detail(exc)}") from None


def _validate_report(report: dict[str, Any]) -> None:
    """Reject a report whose nested shape replay would read unsafely.

    Replay treats the report as recorded data, so a hash-valid package with a
    malformed report must fail closed here instead of surfacing as a
    ``TypeError``/``AttributeError`` while comparing. Nothing is coerced or
    repaired: anything replay cannot read raises ``PackageError``.
    """
    dimensions = report.get("dimensions")
    if dimensions is None:
        return
    if not isinstance(dimensions, list):
        raise PackageError("report.json dimensions must be a list")
    for entry in dimensions:
        if not isinstance(entry, dict):
            raise PackageError("report.json dimensions entries must be objects")
        if not isinstance(entry.get("dimension"), str):
            raise PackageError("report.json dimension entry has no dimension name")
        metrics = entry.get("metrics")
        if metrics is not None and not isinstance(metrics, dict):
            raise PackageError("report.json dimension metrics must be an object or null")
        findings = entry.get("findings")
        if findings is not None and not isinstance(findings, list):
            raise PackageError("report.json dimension findings must be a list")
        error = entry.get("error")
        if error is not None and not isinstance(error, str):
            raise PackageError("report.json dimension error must be a string or null")
        if isinstance(metrics, dict):
            active = metrics.get("active")
            if active is not None and not isinstance(active, dict):
                raise PackageError("report.json dimension metrics.active must be an object or null")


def read_grader_package(archive: Path) -> GraderPackage:
    with tempfile.TemporaryDirectory(prefix="eva-replay-") as temporary:
        workdir = Path(temporary) / "package"
        manifest = extract_zip(archive, workdir)
        _require_grader_package(manifest)
        plan = _load_member(workdir, "plan.json", EvalPlan.model_validate_json)
        inputs = _load_member(workdir, "input.json", PackageInput.model_validate_json)
        snapshots = _load_member(workdir, "snapshots/gradings.json", SnapshotSet.model_validate_json)
        report = _load_member(workdir, "report.json", json.loads)
        if not isinstance(report, dict):
            raise PackageError("report.json is not an object")
        _validate_report(report)
    return GraderPackage(manifest=manifest, plan=plan, inputs=inputs, snapshots=snapshots, report=report)
