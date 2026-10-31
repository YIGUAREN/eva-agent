import json
import shutil
import zipfile

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.domain.run import RunStatus
from eva_agent.packages import run_bundle as run_bundle_module
from eva_agent.packages.manifest import (
    FileEntry,
    PackageError,
    PackageManifest,
    extract_zip,
    sha256_bytes,
    sha256_file,
    write_manifest,
    write_zip,
)
from eva_agent.packages.run_bundle import build_run_bundle, import_run_bundle
from eva_agent.report.render import REPORT_VERSION
from eva_agent.runtime.artifacts import artifact_file, run_paths
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import ArtifactRow, ReportRow, RunEventCounterRow, RunRow
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import budget, dataset_input, fake_config, paste_input, sample_plan

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


async def _reset_run(_config, run_id: str = "run-1") -> None:
    """Remove the run, its report and its artifacts so an import has a clean slate."""
    async with _factory() as session:
        await session.execute(delete(ArtifactRow).where(ArtifactRow.run_id == run_id))
        await session.execute(delete(ReportRow).where(ReportRow.run_id == run_id))
        await session.execute(delete(RunEventCounterRow).where(RunEventCounterRow.run_id == run_id))
        await session.execute(delete(RunRow).where(RunRow.id == run_id))
        await session.commit()
    shutil.rmtree(run_paths(_config.storage.data_dir, run_id).root, ignore_errors=True)


async def _completed_dataset_run(tmp_path):
    """A public, completed DATASET run whose input points at an absolute CSV path."""
    global _factory
    config = fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    _factory = create_session_factory(engine)
    runs = RunRepository(_factory)
    source = tmp_path / "data.csv"
    source.write_text("question,answer\nq,a\n", encoding="utf-8")
    await runs.create("run-ds", dataset_input(source), budget())
    await runs.set_plan("run-ds", sample_plan())
    await runs.set_status("run-ds", RunStatus.COMPLETED)
    run_row = await runs.set_public("run-ds", True)
    await ReportRepository(_factory).upsert(
        "run-ds", REPORT_VERSION, {"version": REPORT_VERSION}, "# report\n"
    )
    return config, run_row


def _rewrite_run_document(tmp_path, archive, mutate, *, report_version: int | None = None):
    """Rebuild ``archive`` after mutating its ``run.json``, rehashing the member."""
    root = tmp_path / "rewritten"
    manifest = extract_zip(archive, root)
    document = json.loads((root / "run.json").read_text(encoding="utf-8"))
    mutate(document)
    data = json.dumps(document).encode("utf-8")
    (root / "run.json").write_bytes(data)
    files = {
        **manifest.files,
        "run.json": FileEntry(sha256=sha256_bytes(data), size=len(data)),
    }
    updated = manifest.model_copy(
        update={
            "files": files,
            "report_version": (
                manifest.report_version if report_version is None else report_version
            ),
        }
    )
    write_manifest(root, updated)
    return write_zip(root, tmp_path / "rewritten.run-bundle.zip", manifest=updated)


async def _assert_nothing_written(config, run_id: str) -> None:
    """No row and no run directory may exist for a rejected import."""
    async with _factory() as session:
        assert await session.get(RunRow, run_id) is None
        assert await session.get(ReportRow, run_id) is None
        assert (
            await session.scalars(select(ArtifactRow).where(ArtifactRow.run_id == run_id))
        ).all() == []
        assert (
            await session.scalars(
                select(RunEventCounterRow).where(RunEventCounterRow.run_id == run_id)
            )
        ).all() == []
    assert not run_paths(config.storage.data_dir, run_id).root.exists()


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


async def _archive_for_completed_run(tmp_path):
    """Build an archive for ``run-1`` and remove the source run, ready for import."""
    config, run_row = await _completed_run(tmp_path)
    report = await ReportRepository(_factory).get("run-1")
    artifacts = await ArtifactRepository(_factory).list_public("run-1")
    archive, _ = await build_run_bundle(
        config=config, run_row=run_row, report=report, artifacts=artifacts, out_dir=tmp_path / "out"
    )
    await _reset_run(config)
    return config, archive


async def _archive_for_dataset_run(tmp_path):
    config, run_row = await _completed_dataset_run(tmp_path)
    report = await ReportRepository(_factory).get("run-ds")
    archive, _ = await build_run_bundle(
        config=config, run_row=run_row, report=report, artifacts=[], out_dir=tmp_path / "out"
    )
    return config, archive


async def test_failed_import_rolls_back_and_a_retry_imports(tmp_path, monkeypatch) -> None:
    config, archive = await _archive_for_completed_run(tmp_path)
    # Fail after the artifacts and report are written, so the rollback must remove
    # both database rows and files.
    original_set_status = RunRepository.set_status
    attempts = 0

    async def flaky_set_status(self, run_id, status):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected import failure")
        await original_set_status(self, run_id, status)

    monkeypatch.setattr(RunRepository, "set_status", flaky_set_status)

    with pytest.raises(RuntimeError, match="injected import failure"):
        await import_run_bundle(config=config, archive=archive, public=True)

    assert await RunRepository(_factory).get("run-1") is None
    assert await ReportRepository(_factory).get("run-1") is None
    await _assert_nothing_written(config, "run-1")

    result = await import_run_bundle(config=config, archive=archive, public=True)

    assert result.status == "imported"
    restored = await RunRepository(_factory).get("run-1")
    assert restored is not None and restored.status == RunStatus.COMPLETED.value


async def test_import_treats_a_concurrent_create_race_as_skipped(tmp_path, monkeypatch) -> None:
    config, archive = await _archive_for_completed_run(tmp_path)
    discards: list[str] = []

    async def losing_create(self, run_id, run_input, budget, **kwargs):
        raise IntegrityError("INSERT INTO runs", {}, Exception("UNIQUE constraint failed: runs.id"))

    async def record_discard(factory, config, run_id):
        discards.append(run_id)

    monkeypatch.setattr(RunRepository, "create", losing_create)
    monkeypatch.setattr(run_bundle_module, "_discard_import", record_discard)

    result = await import_run_bundle(config=config, archive=archive, public=True)

    assert result.status == "skipped"
    # The rows the winner created are not ours to delete.
    assert discards == []


async def test_import_validation_error_redacts_a_path_like_key(tmp_path) -> None:
    config, archive = await _archive_for_dataset_run(tmp_path)
    await _reset_run(config, "run-ds")
    doc_changed = _rewrite_run_document(
        tmp_path,
        archive,
        lambda document: document["input"]["dataset"].__setitem__(
            "column_map", {"C:\\secret\\x.csv": 5}
        ),
    )

    with pytest.raises(PackageError) as caught:
        await import_run_bundle(config=config, archive=doc_changed, public=True)
    message = str(caught.value)
    assert "C:\\secret" not in message
    assert "secret" not in message
    assert "dataset.column_map.<redacted>" in message
    await _assert_nothing_written(config, "run-ds")


async def test_import_rejects_an_out_of_range_report_version(tmp_path) -> None:
    config, archive = await _archive_for_completed_run(tmp_path)
    doc_changed = _rewrite_run_document(
        tmp_path,
        archive,
        lambda document: document.__setitem__("report_version", 2**31),
        report_version=2**31,
    )

    with pytest.raises(PackageError, match="below 2\\^31"):
        await import_run_bundle(config=config, archive=doc_changed, public=True)
    await _assert_nothing_written(config, "run-1")


async def test_import_rejects_a_non_list_artifacts_container(tmp_path) -> None:
    config, archive = await _archive_for_completed_run(tmp_path)
    doc_changed = _rewrite_run_document(
        tmp_path, archive, lambda document: document.pop("artifacts")
    )

    with pytest.raises(PackageError, match="artifacts must be a list"):
        await import_run_bundle(config=config, archive=doc_changed, public=True)
    await _assert_nothing_written(config, "run-1")


@pytest.mark.parametrize("name", ["report.json", "REPORT.JSON"])
async def test_import_rejects_artifact_names_that_shadow_run_files(tmp_path, name) -> None:
    config, archive = await _archive_for_completed_run(tmp_path)
    doc_changed = _rewrite_run_document(
        tmp_path,
        archive,
        lambda document: document.__setitem__(
            "artifacts", [{"name": name, "content_type": "application/json"}]
        ),
    )

    with pytest.raises(PackageError, match="shadows a run file"):
        await import_run_bundle(config=config, archive=doc_changed, public=True)
    await _assert_nothing_written(config, "run-1")


async def test_import_rejects_a_traversal_artifact_name_without_writing(tmp_path) -> None:
    config, archive = await _archive_for_completed_run(tmp_path)
    doc_changed = _rewrite_run_document(
        tmp_path,
        archive,
        lambda document: document.__setitem__(
            "artifacts", [{"name": "../escape.txt", "content_type": "text/plain"}]
        ),
    )

    with pytest.raises(PackageError, match="unsafe artifact name"):
        await import_run_bundle(config=config, archive=doc_changed, public=True)
    await _assert_nothing_written(config, "run-1")


async def test_import_rejects_a_missing_artifact_member_without_writing(tmp_path) -> None:
    config, archive = await _archive_for_completed_run(tmp_path)
    doc_changed = _rewrite_run_document(
        tmp_path,
        archive,
        lambda document: document.__setitem__(
            "artifacts", [{"name": "missing.txt", "content_type": "text/plain"}]
        ),
    )

    with pytest.raises(PackageError, match="missing artifacts/missing.txt"):
        await import_run_bundle(config=config, archive=doc_changed, public=True)
    await _assert_nothing_written(config, "run-1")


async def test_import_does_not_overwrite_an_existing_run(tmp_path) -> None:
    config, archive = await _archive_for_completed_run(tmp_path)
    first = await import_run_bundle(config=config, archive=archive, public=True)
    assert first.status == "imported"
    await ReportRepository(_factory).upsert(
        "run-1", REPORT_VERSION, {"version": REPORT_VERSION, "changed": True}, "# changed\n"
    )
    await RunRepository(_factory).set_status("run-1", RunStatus.FAILED)

    second = await import_run_bundle(config=config, archive=archive, public=True)

    assert second.status == "skipped"
    stored = await ReportRepository(_factory).get("run-1")
    assert stored is not None and stored.payload.get("changed") is True
    restored = await RunRepository(_factory).get("run-1")
    assert restored is not None and restored.status == RunStatus.FAILED.value


async def test_import_with_public_false_keeps_the_run_private(tmp_path) -> None:
    config, archive = await _archive_for_completed_run(tmp_path)

    result = await import_run_bundle(config=config, archive=archive, public=False)

    assert result.status == "imported"
    restored = await RunRepository(_factory).get("run-1")
    assert restored is not None
    assert restored.is_public is False
    assert restored.published_at is None


async def test_dataset_bundle_round_trips_with_a_basename(tmp_path) -> None:
    config, archive = await _archive_for_dataset_run(tmp_path)
    with zipfile.ZipFile(archive) as bundle:
        document = json.loads(bundle.read("run.json"))
    assert document["input"]["dataset"]["path"] == "data.csv"
    await _reset_run(config, "run-ds")

    result = await import_run_bundle(config=config, archive=archive, public=True)

    assert result.status == "imported"
    restored = await RunRepository(_factory).get("run-ds")
    assert restored is not None
    assert restored.input_json["dataset"]["path"] == "data.csv"


async def test_import_rejects_an_absolute_dataset_path(tmp_path) -> None:
    config, archive = await _archive_for_dataset_run(tmp_path)
    doc_changed = _rewrite_run_document(
        tmp_path,
        archive,
        lambda document: document["input"]["dataset"].__setitem__(
            "path", "C:\\secret\\x.csv"
        ),
    )
    await _reset_run(config, "run-ds")

    with pytest.raises(PackageError, match="plain file name"):
        await import_run_bundle(config=config, archive=doc_changed, public=True)
    await _assert_nothing_written(config, "run-ds")
