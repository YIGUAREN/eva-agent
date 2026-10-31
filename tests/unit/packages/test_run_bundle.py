import json
import zipfile

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.domain.run import RunStatus
from eva_agent.packages import run_bundle as run_bundle_module
from eva_agent.packages.manifest import (
    FileEntry,
    PackageError,
    PackageManifest,
    sha256_bytes,
    sha256_file,
    write_manifest,
    write_zip,
)
from eva_agent.packages.run_bundle import build_run_bundle, import_run_bundle
from eva_agent.report.render import REPORT_VERSION
from eva_agent.runtime.artifacts import artifact_file, run_paths
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import ArtifactRow, ReportRow, RunRow
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import budget, fake_config, paste_input, sample_plan

_factory: async_sessionmaker[AsyncSession]


async def _completed_run(tmp_path):
    """A public, completed run with a stored report and one public artifact on disk."""
    global _factory
    config = fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    _factory = create_session_factory(engine)
    runs = RunRepository(_factory)
    await runs.create("run-1", paste_input(), budget())
    await runs.set_plan("run-1", sample_plan())
    await runs.set_status("run-1", RunStatus.COMPLETED)
    run_row = await runs.set_public("run-1", True)
    await ReportRepository(_factory).upsert(
        "run-1", REPORT_VERSION, {"version": REPORT_VERSION}, "# report\n"
    )
    target = artifact_file(config.storage.data_dir, "run-1", "public.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"ok": true}', encoding="utf-8")
    await ArtifactRepository(_factory).add(
        artifact_id="art-public",
        run_id="run-1",
        name="public.json",
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
        content_type="application/json",
        visibility="public",
    )
    return config, run_row


async def _reset_run(_config) -> None:
    """Remove the run, its report and its artifacts so an import has a clean slate."""
    async with _factory() as session:
        await session.execute(delete(ArtifactRow).where(ArtifactRow.run_id == "run-1"))
        await session.execute(delete(ReportRow).where(ReportRow.run_id == "run-1"))
        await session.execute(delete(RunRow).where(RunRow.id == "run-1"))
        await session.commit()


def _bundle_with_run_id(tmp_path, run_id: str):
    """A hash-valid bundle whose manifest and run.json carry ``run_id``."""
    root = tmp_path / "traversal-bundle"
    root.mkdir()
    members = {
        "run.json": json.dumps({"run_id": run_id, "status": "completed"}).encode("utf-8"),
        "report.json": b"{}",
        "report.md": b"",
    }
    for name, data in members.items():
        (root / name).write_bytes(data)
    manifest = PackageManifest(
        kind="run_bundle",
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        report_version=REPORT_VERSION,
        replayable=False,
        files={
            name: FileEntry(sha256=sha256_file(root / name), size=len(data))
            for name, data in members.items()
        },
    )
    write_manifest(root, manifest)
    return write_zip(root, tmp_path / "traversal.run-bundle.zip", manifest=manifest)


async def _published_run(write_run):
    """The conftest run, published, with a stored report and two artifacts on disk."""
    config, run_row = await write_run()
    factory = create_session_factory(create_engine(config.storage.url))
    paths = run_paths(config.storage.data_dir, run_row.id)
    payload = json.loads(paths.report_json.read_text(encoding="utf-8"))
    payload = {**payload, "api_key": "sk-live-123"}
    markdown = f"# 报告\n- 报告文件：{paths.report_json}\n"
    # ``set_public`` returns the freshly loaded row; the fixture's row predates publication.
    run_row = await RunRepository(factory).set_public(run_row.id, True)
    await ReportRepository(factory).upsert(run_row.id, REPORT_VERSION, payload, markdown)
    artifacts = ArtifactRepository(factory)
    for name, visibility in (("public.json", "public"), ("secret.json", "internal")):
        target = artifact_file(config.storage.data_dir, run_row.id, name)
        target.write_text("{}", encoding="utf-8")
        await artifacts.add(
            artifact_id=f"art-{name}",
            run_id=run_row.id,
            name=name,
            sha256="a" * 64,
            size_bytes=2,
            content_type="application/json",
            visibility=visibility,
        )
    return config, run_row, factory


async def test_bundle_carries_only_public_artifacts(write_run, tmp_path) -> None:
    config, run_row, factory = await _published_run(write_run)
    report = await ReportRepository(factory).get(run_row.id)
    artifacts = await ArtifactRepository(factory).list_public(run_row.id)
    archive, manifest = await build_run_bundle(
        config=config,
        run_row=run_row,
        report=report,
        artifacts=artifacts,
        out_dir=tmp_path / "out",
    )
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert manifest.kind == "run_bundle"
    assert manifest.replayable is False
    assert {"manifest.json", "run.json", "report.json", "report.md"} <= names
    assert "artifacts/public.json" in names
    assert not any("secret" in name for name in names)


async def test_bundle_redacts_paths_and_secrets(write_run, tmp_path) -> None:
    config, run_row, factory = await _published_run(write_run)
    report = await ReportRepository(factory).get(run_row.id)
    artifacts = await ArtifactRepository(factory).list_public(run_row.id)
    archive, _ = await build_run_bundle(
        config=config,
        run_row=run_row,
        report=report,
        artifacts=artifacts,
        out_dir=tmp_path / "out",
    )
    with zipfile.ZipFile(archive) as bundle:
        markdown = bundle.read("report.md").decode("utf-8")
        payload = json.loads(bundle.read("report.json"))
    assert str(config.storage.data_dir) not in markdown
    assert "api_key" not in payload
    assert "sk-live-123" not in json.dumps(payload)


async def test_import_restores_the_public_run(tmp_path) -> None:
    config, run_row = await _completed_run(tmp_path)
    report = await ReportRepository(_factory).get("run-1")
    artifacts = await ArtifactRepository(_factory).list_public("run-1")
    archive, _ = await build_run_bundle(
        config=config, run_row=run_row, report=report, artifacts=artifacts, out_dir=tmp_path / "out"
    )
    await _reset_run(config)

    result = await import_run_bundle(config=config, archive=archive, public=True)

    assert result.status == "imported"
    assert result.artifacts == 1
    restored = await RunRepository(_factory).get("run-1")
    assert restored is not None
    assert restored.status == RunStatus.COMPLETED.value
    assert restored.is_public is True
    stored = await ReportRepository(_factory).get("run-1")
    assert stored is not None and report is not None and stored.version == report.version
    assert (await ArtifactRepository(_factory).list_public("run-1"))[0].name == "public.json"


async def test_import_is_idempotent_by_run_id(tmp_path) -> None:
    config, run_row = await _completed_run(tmp_path)
    report = await ReportRepository(_factory).get("run-1")
    artifacts = await ArtifactRepository(_factory).list_public("run-1")
    archive, _ = await build_run_bundle(
        config=config, run_row=run_row, report=report, artifacts=artifacts, out_dir=tmp_path / "out"
    )
    await _reset_run(config)
    first = await import_run_bundle(config=config, archive=archive, public=True)
    second = await import_run_bundle(config=config, archive=archive, public=True)
    assert (first.status, second.status) == ("imported", "skipped")


async def test_import_rejects_a_traversal_run_id(tmp_path) -> None:
    archive = _bundle_with_run_id(tmp_path, "../evil")
    with pytest.raises(PackageError):
        await import_run_bundle(config=fake_config(tmp_path), archive=archive, public=True)


async def test_builder_filters_non_public_artifacts_passed_directly(tmp_path) -> None:
    config, run_row = await _completed_run(tmp_path)
    report = await ReportRepository(_factory).get("run-1")
    repository = ArtifactRepository(_factory)
    target = artifact_file(config.storage.data_dir, "run-1", "secret.txt")
    target.write_text("internal only", encoding="utf-8")
    await repository.add(
        artifact_id="art-secret",
        run_id="run-1",
        name="secret.txt",
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
        content_type="text/plain",
        visibility="internal",
    )
    secret = await repository.get("run-1", "secret.txt")
    assert secret is not None
    # Feed an unfiltered list: the builder must re-filter ``visibility`` itself.
    artifacts = [*(await repository.list_public("run-1")), secret]

    archive, manifest = await build_run_bundle(
        config=config,
        run_row=run_row,
        report=report,
        artifacts=artifacts,
        out_dir=tmp_path / "out",
    )

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        run_document = json.loads(bundle.read("run.json"))
    assert "artifacts/secret.txt" not in names
    assert "artifacts/secret.txt" not in manifest.files
    assert all(item["name"] != "secret.txt" for item in run_document["artifacts"])


async def test_bundle_redacts_the_data_dir_in_isolation(tmp_path) -> None:
    config, run_row = await _completed_run(tmp_path)
    # A word character directly before the path stops ``redact_markdown_paths``
    # (it only matches at a token boundary), so only the data-dir replacement can
    # remove this occurrence.
    markdown = f"prefix{config.storage.data_dir}suffix"
    await ReportRepository(_factory).upsert(
        "run-1", REPORT_VERSION, {"version": REPORT_VERSION}, markdown
    )
    report = await ReportRepository(_factory).get("run-1")

    archive, _ = await build_run_bundle(
        config=config,
        run_row=run_row,
        report=report,
        artifacts=[],
        out_dir=tmp_path / "out",
    )

    with zipfile.ZipFile(archive) as bundle:
        rendered = bundle.read("report.md").decode("utf-8")
    assert "<data>" in rendered
    assert str(config.storage.data_dir) not in rendered


async def test_bundle_manifest_matches_the_archive_members(tmp_path) -> None:
    config, run_row = await _completed_run(tmp_path)
    report = await ReportRepository(_factory).get("run-1")
    artifacts = await ArtifactRepository(_factory).list_public("run-1")
    archive, manifest = await build_run_bundle(
        config=config,
        run_row=run_row,
        report=report,
        artifacts=artifacts,
        out_dir=tmp_path / "out",
    )

    with zipfile.ZipFile(archive) as bundle:
        members = {
            name: bundle.read(name)
            for name in bundle.namelist()
            if name != "manifest.json"
        }
    assert set(manifest.files) == set(members)
    for name, data in members.items():
        entry = manifest.files[name]
        assert entry.size == len(data)
        assert entry.sha256 == sha256_bytes(data)


async def test_builder_verifies_the_staged_tree(tmp_path, monkeypatch) -> None:
    config, run_row = await _completed_run(tmp_path)
    report = await ReportRepository(_factory).get("run-1")
    artifacts = await ArtifactRepository(_factory).list_public("run-1")
    calls: list[tuple[object, PackageManifest]] = []
    monkeypatch.setattr(
        run_bundle_module, "verify_tree", lambda root, manifest: calls.append((root, manifest))
    )

    await build_run_bundle(
        config=config,
        run_row=run_row,
        report=report,
        artifacts=artifacts,
        out_dir=tmp_path / "out",
    )

    assert len(calls) == 1
    assert calls[0][1].run_id == "run-1"


async def test_failed_archive_write_leaves_the_previous_bundle_intact(
    tmp_path, monkeypatch
) -> None:
    config, run_row = await _completed_run(tmp_path)
    report = await ReportRepository(_factory).get("run-1")
    artifacts = await ArtifactRepository(_factory).list_public("run-1")
    out_dir = tmp_path / "out"
    archive, _ = await build_run_bundle(
        config=config,
        run_row=run_row,
        report=report,
        artifacts=artifacts,
        out_dir=out_dir,
    )
    original = archive.read_bytes()

    def failing_write(root, destination, *, manifest):
        destination.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(run_bundle_module, "write_zip", failing_write)
    with pytest.raises(OSError, match="disk full"):
        await build_run_bundle(
            config=config,
            run_row=run_row,
            report=report,
            artifacts=artifacts,
            out_dir=out_dir,
        )

    assert archive.read_bytes() == original
    assert not [path for path in out_dir.iterdir() if ".tmp-" in path.name]


async def test_builder_rejects_an_invalid_run_id(tmp_path) -> None:
    config, run_row = await _completed_run(tmp_path)
    report = await ReportRepository(_factory).get("run-1")
    run_row.id = "../evil"

    with pytest.raises(ValueError, match="invalid run_id"):
        await build_run_bundle(
            config=config,
            run_row=run_row,
            report=report,
            artifacts=[],
            out_dir=tmp_path / "out",
        )
