"""Export a public run as a portable bundle, and import one back.

A bundle is an admin-level artifact, equivalent to the admin report endpoint
rather than the public API whitelist: it deliberately carries the run's spend
(``usage.spent_usd`` in ``report.json``) and budget (``budget_json`` in
``run.json``) so ``import-run`` can restore the run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.config.settings import AppConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.run import RunInput, RunStatus, validate_run_id
from eva_agent.packages.manifest import (
    MAX_PACKAGE_FILES,
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
from eva_agent.runtime.artifacts import artifact_file
from eva_agent.server.schemas import (
    public_run_detail,
    sanitize_public_markdown,
    sanitize_public_payload,
)
from eva_agent.storage.db import create_engine, create_session_factory
from eva_agent.storage.models import ArtifactRow, RunRow
from eva_agent.storage.reports import ArtifactRepository, ReportRepository, ReportSnapshot
from eva_agent.storage.repository import RunRepository

ARCHIVE_SUFFIX = ".run-bundle.zip"


def _internal_hosts(config: AppConfig) -> tuple[str, ...]:
    """Return the configured AIG host so the public serializers can redact it.

    Deliberately mirrors ``_internal_hosts`` in ``server.routes.runs``: importing
    the HTTP routes layer from ``packages`` would drag the server into bundle
    code, and no neutral shared module owns this config-to-host mapping yet.
    """
    aig = config.aig
    if aig is None:
        return ()
    try:
        hostname = urlsplit(aig.base_url).hostname
    except ValueError:
        return ()
    return (hostname,) if hostname else ()


def _safe_input(payload: dict) -> dict:
    """Reduce a stored run input to a path-free copy.

    This mirrors ``_safe_input`` in ``server.schemas``. It is duplicated here
    rather than imported because that one is private to the server package.
    """
    data = dict(payload)
    dataset = data.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("path"), str):
        data["dataset"] = {**dataset, "path": Path(dataset["path"]).name}
    return data


def _run_document(
    run_row: RunRow,
    report: ReportSnapshot,
    artifacts: list[ArtifactRow],
    *,
    internal_hosts: tuple[str, ...] = (),
) -> dict:
    detail = public_run_detail(run_row, internal_hosts=internal_hosts)
    return {
        "run_id": run_row.id,
        "status": run_row.status,
        "created_at": detail["created_at"],
        "objective": detail["objective"],
        "input": sanitize_public_payload(
            _safe_input(run_row.input_json or {}), internal_hosts=internal_hosts
        ),
        "budget": sanitize_public_payload(run_row.budget_json, internal_hosts=internal_hosts),
        "plan": sanitize_public_payload(run_row.plan_json, internal_hosts=internal_hosts),
        "report_version": report.version,
        "artifacts": [
            {"name": row.name, "content_type": row.content_type} for row in artifacts
        ],
    }


async def build_run_bundle(
    *,
    config: AppConfig,
    run_row: RunRow,
    report: ReportSnapshot | None,
    artifacts: list[ArtifactRow],
    out_dir: Path,
) -> tuple[Path, PackageManifest]:
    # Validate up front so the archive name and the manifest run_id are
    # self-contained, rather than only being checked incidentally per artifact.
    validate_run_id(run_row.id)
    if run_row.status != RunStatus.COMPLETED.value:
        raise PackageError("only completed runs can be exported")
    if not run_row.is_public:
        raise PackageError("only public runs can be exported")
    if report is None:
        raise PackageError("run has no stored report")
    # Defense in depth: the caller normally passes ``ArtifactRepository.list_public``
    # output, but a non-public artifact must never reach the bundle regardless.
    public_artifacts = [row for row in artifacts if row.visibility == "public"]
    hosts = _internal_hosts(config)
    workdir = Path(tempfile.mkdtemp(prefix="eva-bundle-"))
    try:
        (workdir / "artifacts").mkdir(parents=True)
        (workdir / "run.json").write_bytes(
            canonical_json(_run_document(run_row, report, public_artifacts, internal_hosts=hosts))
        )
        (workdir / "report.json").write_bytes(
            canonical_json(sanitize_public_payload(report.payload, internal_hosts=hosts))
        )
        (workdir / "report.md").write_text(
            sanitize_public_markdown(
                report.markdown, data_dir=config.storage.data_dir, internal_hosts=hosts
            ),
            encoding="utf-8",
        )
        members = ["run.json", "report.json", "report.md"]
        seen: set[str] = set()
        for row in public_artifacts:
            basename = Path(row.name).name
            if basename in seen:
                raise PackageError(f"artifact basename collision for {row.name}")
            seen.add(basename)
            # Only the basename enters the bundle, so an AIG path such as
            # aig/session.json can never escape the artifacts/ directory.
            relative = f"artifacts/{basename}"
            source = artifact_file(config.storage.data_dir, run_row.id, row.name)
            if not source.is_file():
                raise PackageError(f"artifact {row.name} is missing on disk")
            target = workdir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            members.append(relative)
        files = {
            name: FileEntry(
                sha256=sha256_file(workdir / name), size=(workdir / name).stat().st_size
            )
            for name in members
        }
        manifest = PackageManifest(
            kind="run_bundle",
            run_id=run_row.id,
            created_at=datetime.now(UTC).isoformat(),
            report_version=report.version,
            replayable=False,
            replay_blockers=[
                "a run bundle carries no grading inputs; export a grader package to replay"
            ],
            files=files,
        )
        write_manifest(workdir, manifest)
        verify_tree(workdir, manifest)
        out_dir.mkdir(parents=True, exist_ok=True)
        destination = out_dir / f"{run_row.id}{ARCHIVE_SUFFIX}"
        # Write beside the destination and move it into place only on success,
        # so a mid-write failure cannot truncate a previous bundle with the same
        # name. This mirrors the staging pattern in ``manifest.extract_zip``.
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


@dataclass(frozen=True)
class ImportResult:
    run_id: str
    status: Literal["imported", "skipped"]
    artifacts: int = 0


@dataclass(frozen=True)
class _RestoreArtifact:
    name: str
    content_type: str
    source: Path
    target: Path


_ARTIFACT_SAFE = re.compile(r"[A-Za-z0-9._-]{1,255}")
_MAX_CONTENT_TYPE = 128


def _reason(exc: BaseException) -> str:
    """A path-free error detail; ``str(OSError)`` embeds the absolute path."""
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return str(exc) or type(exc).__name__


def _validation_message(exc: ValidationError) -> str:
    """Summarize a validation failure without echoing the offending input value."""
    details = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    )
    return f"bundle payload is invalid: {details}"


def _read_json(path: Path, label: str) -> object:
    if not path.is_file():
        raise PackageError(f"bundle is missing {label}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"cannot read {label}: {_reason(exc)}") from None


def _read_bundle(workdir: Path) -> dict:
    document = _read_json(workdir / "run.json", "run.json")
    if not isinstance(document, dict):
        raise PackageError("run.json must be an object")
    if document.get("status") != RunStatus.COMPLETED.value:
        raise PackageError("only completed runs can be imported")
    return document


def _payload(document: dict, key: str) -> dict:
    value = document.get(key)
    if not isinstance(value, dict):
        raise PackageError(f"run.json is missing a {key} object")
    return value


def _report_version(document: dict) -> int:
    value = document.get("report_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PackageError("report_version must be a non-negative integer")
    return value


def _read_report(workdir: Path) -> tuple[dict, str]:
    payload = _read_json(workdir / "report.json", "report.json")
    if not isinstance(payload, dict):
        raise PackageError("report.json must be an object")
    markdown_path = workdir / "report.md"
    if not markdown_path.is_file():
        raise PackageError("bundle is missing report.md")
    try:
        return payload, markdown_path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise PackageError(f"cannot read report.md: {_reason(exc)}") from None


def _artifact_name(name: object) -> str:
    if not isinstance(name, str):
        raise PackageError("artifact name must be a string")
    if _ARTIFACT_SAFE.fullmatch(name) is not None:
        if name in {".", ".."}:
            raise PackageError("unsafe artifact name")
        return name
    if name.startswith("aig/"):
        # ``artifact_file`` applies the stricter ``aig/<id>.json|csv`` shape.
        return name
    raise PackageError("unsafe artifact name")


def _restore_artifacts(
    config: AppConfig, run_id: str, document: dict, workdir: Path
) -> list[_RestoreArtifact]:
    """Validate every artifact entry and member before the import writes anything."""
    raw = document.get("artifacts") or []
    if not isinstance(raw, list):
        raise PackageError("artifacts must be a list")
    if len(raw) > MAX_PACKAGE_FILES:
        raise PackageError(f"bundle has {len(raw)} artifacts; the limit is {MAX_PACKAGE_FILES}")
    entries: list[_RestoreArtifact] = []
    names: set[str] = set()
    basenames: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise PackageError("artifact entries must be objects")
        name = _artifact_name(entry.get("name"))
        if name in names:
            raise PackageError(f"duplicate artifact name {name!r}")
        names.add(name)
        try:
            target = artifact_file(config.storage.data_dir, run_id, name)
        except ValueError:
            raise PackageError("unsafe artifact name") from None
        basename = Path(name).name
        if basename in basenames:
            raise PackageError(f"artifact member collision for {name!r}")
        basenames.add(basename)
        source = workdir / "artifacts" / basename
        if not source.is_file():
            raise PackageError(f"bundle is missing artifacts/{basename}")
        content_type = entry.get("content_type") or "application/octet-stream"
        if (
            not isinstance(content_type, str)
            or not content_type
            or len(content_type) > _MAX_CONTENT_TYPE
            or "\r" in content_type
            or "\n" in content_type
        ):
            raise PackageError(f"invalid content_type for artifact {name!r}")
        entries.append(_RestoreArtifact(name, content_type, source, target))
    return entries


async def _write_artifacts(
    artifacts: ArtifactRepository, run_id: str, entries: list[_RestoreArtifact]
) -> int:
    for entry in entries:
        try:
            entry.target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry.source, entry.target)
            digest = sha256_file(entry.target)
            size = entry.target.stat().st_size
        except OSError as exc:
            raise PackageError(f"cannot write artifact: {_reason(exc)}") from None
        await artifacts.add(
            artifact_id=f"art_{uuid4().hex}",
            run_id=run_id,
            name=entry.name,
            sha256=digest,
            size_bytes=size,
            content_type=entry.content_type,
            visibility="public",
        )
    return len(entries)


@asynccontextmanager
async def _storage(
    config: AppConfig,
) -> AsyncIterator[tuple[RunRepository, ReportRepository, ArtifactRepository]]:
    engine = create_engine(config.storage.url)
    try:
        factory: async_sessionmaker[AsyncSession] = create_session_factory(engine)
        yield RunRepository(factory), ReportRepository(factory), ArtifactRepository(factory)
    finally:
        await engine.dispose()


async def import_run_bundle(*, config: AppConfig, archive: Path, public: bool) -> ImportResult:
    """Restore a completed run from ``archive``; an existing ``run_id`` is skipped."""
    workdir = Path(tempfile.mkdtemp(prefix="eva-import-"))
    try:
        manifest = extract_zip(archive, workdir)
        if manifest.kind != "run_bundle":
            raise PackageError(f"not a run bundle: {manifest.kind}")
        run_id = manifest.run_id
        try:
            validate_run_id(run_id)
        except ValueError:
            raise PackageError("bundle run_id is invalid") from None
        document = _read_bundle(workdir)
        if document.get("run_id") != run_id:
            raise PackageError("run.json does not match the manifest run_id")
        raw_budget = _payload(document, "budget")
        # The exporter runs the plan through ``sanitize_public_payload``, which
        # drops any nested key named ``budget``.  Re-attach the run's budget so
        # the plan we wrote can validate again.
        raw_plan = dict(_payload(document, "plan"))
        raw_plan.setdefault("budget", raw_budget)
        try:
            run_input = RunInput.model_validate(_safe_input(_payload(document, "input")))
            budget = Budget.model_validate(raw_budget)
            plan = EvalPlan.model_validate(raw_plan)
        except ValidationError as exc:
            raise PackageError(_validation_message(exc)) from None
        report_version = _report_version(document)
        if manifest.report_version is not None and manifest.report_version != report_version:
            raise PackageError("report_version does not match the manifest")
        report_payload, report_markdown = _read_report(workdir)
        entries = _restore_artifacts(config, run_id, document, workdir)
        async with _storage(config) as (runs, reports, artifacts):
            if await runs.get(run_id) is not None:
                return ImportResult(run_id=run_id, status="skipped")
            await runs.create(run_id, run_input, budget)
            await runs.set_plan(run_id, plan)
            await reports.upsert(run_id, report_version, report_payload, report_markdown)
            restored = await _write_artifacts(artifacts, run_id, entries)
            await runs.set_status(run_id, RunStatus.COMPLETED)
            if public:
                await runs.set_public(run_id, True)
        return ImportResult(run_id=run_id, status="imported", artifacts=restored)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
