"""Export a public run as a portable bundle."""

from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from eva_agent.config.settings import AppConfig
from eva_agent.domain.run import RunStatus
from eva_agent.packages.manifest import (
    FileEntry,
    PackageError,
    PackageManifest,
    canonical_json,
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
from eva_agent.storage.models import ArtifactRow, RunRow
from eva_agent.storage.reports import ReportSnapshot

ARCHIVE_SUFFIX = ".run-bundle.zip"


def _internal_hosts(config: AppConfig) -> tuple[str, ...]:
    """Return the configured AIG host so the public serializers can redact it."""
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
        archive = write_zip(
            workdir, out_dir / f"{run_row.id}{ARCHIVE_SUFFIX}", manifest=manifest
        )
        return archive, manifest
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
