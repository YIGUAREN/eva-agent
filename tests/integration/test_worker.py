import asyncio
import hashlib
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from eva_agent.domain.run import InputKind, RunInput, RunStatus
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.manager import RunManager
from eva_agent.runtime.uploads import upload_paths
from eva_agent.sandbox.extract import extract_archive
from eva_agent.storage.commands import RunCommandRepository
from eva_agent.storage.uploads import UploadRepository
from eva_agent.worker.runner import RunWorker, _public_error
from tests.support.runtime import budget, dependencies, fake_config, paste_input


async def _worker(tmp_path):
    config = fake_config(tmp_path)
    manager = RunManager(config, dependencies=dependencies(tmp_path))
    await manager.setup()
    worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir, poll_interval_s=0.01)
    return manager, worker


async def test_start_then_approve_completes_the_run(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await worker.commands.enqueue(run_id, "start")

    assert await worker.tick() is True
    row = await worker.runs.get(run_id)
    assert row is not None and row.status == RunStatus.WAITING_APPROVAL.value

    await worker.commands.enqueue(run_id, "approve", {"approved": True})
    assert await worker.tick() is True
    row = await worker.runs.get(run_id)
    assert row is not None and row.status == RunStatus.COMPLETED.value

    snapshot = await worker.reports.get(run_id)
    assert snapshot is not None and snapshot.payload["run_id"] == run_id
    report_artifact = await worker.artifacts.get(run_id, "report.json")
    assert report_artifact is not None and report_artifact.visibility == "admin"
    assert await worker.artifacts.get(run_id, "plan.approved.yaml") is not None
    await manager.close()


async def test_cancel_command_stops_a_waiting_run(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await worker.commands.enqueue(run_id, "start")
    await worker.tick()

    await worker.commands.enqueue(run_id, "cancel")
    assert await worker.tick() is True

    row = await worker.runs.get(run_id)
    assert row is not None and row.status == RunStatus.CANCELLED.value
    await manager.close()


async def test_unknown_command_is_recorded_as_failed(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await worker.commands.enqueue(run_id, "bogus")

    assert await worker.tick() is True

    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "failed"
    assert "unknown command kind" in (commands[-1].error or "")
    await manager.close()


async def test_tick_returns_false_when_the_queue_is_empty(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)

    assert await worker.tick() is False
    await manager.close()


async def test_duplicate_start_is_a_no_op_after_completion(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await worker.commands.enqueue(run_id, "start")
    await worker.tick()
    await worker.commands.enqueue(run_id, "approve", {"approved": True})
    await worker.tick()

    await worker.commands.enqueue(run_id, "start")
    assert await worker.tick() is True
    row = await worker.runs.get(run_id)
    assert row is not None and row.status == RunStatus.COMPLETED.value
    await manager.close()


async def test_worker_failure_does_not_crash_the_loop(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await worker.commands.enqueue(run_id, "resume")

    assert await worker.tick() is True
    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "failed"
    assert "cannot be resumed from status 'queued'" in (commands[-1].error or "")
    await manager.close()


async def test_start_on_a_running_run_without_a_checkpoint_fails_the_run(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    repo = manager._require_repo()
    assert await repo.claim(run_id, from_statuses=(RunStatus.QUEUED,), to_status=RunStatus.RUNNING)
    await worker.commands.enqueue(run_id, "start")

    assert await worker.tick() is True

    row = await worker.runs.get(run_id)
    assert row is not None and row.status == RunStatus.FAILED.value
    assert row.error is not None and "no checkpoint" in row.error
    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "failed"
    assert "no checkpoint" in (commands[-1].error or "")
    await manager.close()


async def test_start_on_a_running_run_calls_resume(tmp_path, monkeypatch) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    repo = manager._require_repo()
    assert await repo.claim(run_id, from_statuses=(RunStatus.QUEUED,), to_status=RunStatus.RUNNING)
    resumed: list[str] = []

    async def fake_resume(rid: str):
        resumed.append(rid)

    monkeypatch.setattr(manager, "resume", fake_resume)
    await worker.commands.enqueue(run_id, "start")

    assert await worker.tick() is True
    assert resumed == [run_id]
    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "done"
    await manager.close()


async def test_resume_is_rejected_for_a_run_that_is_not_running(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await worker.commands.enqueue(run_id, "start")
    await worker.tick()

    await worker.commands.enqueue(run_id, "resume")
    assert await worker.tick() is True

    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "failed"
    assert "cannot be resumed from status" in (commands[-1].error or "")
    row = await worker.runs.get(run_id)
    assert row is not None and row.status == RunStatus.WAITING_APPROVAL.value
    await manager.close()


async def test_duplicate_start_backfills_report_rows_after_a_crash(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await manager.start_existing(run_id)
    await manager.approve(run_id, approved=True)

    # The graph completed and wrote report.json, but the worker "crashed" before
    # _after could persist the report and artifact rows.
    assert await worker.reports.get(run_id) is None
    assert await worker.artifacts.get(run_id, "report.json") is None

    await worker.commands.enqueue(run_id, "start")
    assert await worker.tick() is True

    row = await worker.runs.get(run_id)
    assert row is not None and row.status == RunStatus.COMPLETED.value
    snapshot = await worker.reports.get(run_id)
    assert snapshot is not None and snapshot.payload["run_id"] == run_id
    assert await worker.artifacts.get(run_id, "report.json") is not None
    assert await worker.artifacts.get(run_id, "plan.approved.yaml") is not None
    await manager.close()


async def test_run_forever_survives_a_tick_failure(tmp_path, monkeypatch) -> None:
    manager, worker = await _worker(tmp_path)
    stop = asyncio.Event()

    async def failing_tick() -> bool:
        stop.set()
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(worker, "tick", failing_tick)
    await asyncio.wait_for(worker.run_forever(stop), timeout=1.0)
    await manager.close()


async def test_report_persistence_failure_is_not_leaked_in_the_command_error(tmp_path, monkeypatch) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())

    async def failing_after(rid: str) -> None:
        raise RuntimeError(f"cannot read {tmp_path / 'data' / rid / 'report.json'}")

    monkeypatch.setattr(worker, "_after", failing_after)
    await worker.commands.enqueue(run_id, "start")
    assert await worker.tick() is True

    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "failed"
    assert commands[-1].error == "failed to record the run report"
    await manager.close()


async def test_duplicate_resume_backfills_report_rows_after_a_crash(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await manager.start_existing(run_id)
    await manager.approve(run_id, approved=True)

    # The run was driven by a resume command that committed COMPLETED, but the
    # worker crashed before _after persisted the report and artifact rows, so
    # reset_stale requeued it. Replaying resume must repair the rows.
    row = await worker.runs.get(run_id)
    assert row is not None and row.status == RunStatus.COMPLETED.value
    assert await worker.reports.get(run_id) is None
    assert await worker.artifacts.get(run_id, "report.json") is None

    await worker.commands.enqueue(run_id, "resume")
    assert await worker.tick() is True

    snapshot = await worker.reports.get(run_id)
    assert snapshot is not None and snapshot.payload["run_id"] == run_id
    assert await worker.artifacts.get(run_id, "report.json") is not None
    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "done"
    await manager.close()


async def test_resume_on_a_run_awaiting_approval_fails_with_the_status(tmp_path, monkeypatch) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await worker.commands.enqueue(run_id, "start")
    await worker.tick()
    resumed: list[str] = []

    async def fake_resume(rid: str) -> None:
        resumed.append(rid)

    monkeypatch.setattr(manager, "resume", fake_resume)
    await worker.commands.enqueue(run_id, "resume")
    assert await worker.tick() is True

    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "failed"
    assert "waiting_approval" in (commands[-1].error or "")
    assert resumed == []
    row = await worker.runs.get(run_id)
    assert row is not None and row.status == RunStatus.WAITING_APPROVAL.value
    await manager.close()


async def test_unexpected_error_is_sanitized_before_persisting(tmp_path, monkeypatch) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    secret = r"C:\secret\path"

    async def failing_start(rid: str):
        raise RuntimeError(f"{secret} " + "x" * 600)

    monkeypatch.setattr(manager, "start_existing", failing_start)
    await worker.commands.enqueue(run_id, "start")
    assert await worker.tick() is True

    commands = await worker.commands.list_for_run(run_id)
    error = commands[-1].error or ""
    assert commands[-1].status == "failed"
    assert secret not in error
    assert "secret" not in error
    assert len(error) <= 500
    await manager.close()


def test_public_error_keeps_an_exact_value_error() -> None:
    assert _public_error(ValueError("safe")) == "safe"


def test_public_error_hides_value_error_subclasses() -> None:
    error = json.JSONDecodeError("bad", '{"sk-secret12345": 1}', 1)

    assert _public_error(error) == "command failed"


async def _completed_run(manager) -> str:
    run_id = await manager.create_run(paste_input(), budget())
    await manager.start_existing(run_id)
    await manager.approve(run_id, approved=True)
    return run_id


async def test_transient_report_upsert_failure_is_retried(tmp_path, monkeypatch) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await _completed_run(manager)

    original = worker.reports.upsert
    calls = 0

    async def flaky_upsert(rid: str, version: int, payload: dict, markdown: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary database error")
        await original(rid, version, payload, markdown)

    monkeypatch.setattr(worker.reports, "upsert", flaky_upsert)
    await worker.commands.enqueue(run_id, "start")

    assert await worker.tick() is True
    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "done"
    assert calls == 2
    assert await worker.reports.get(run_id) is not None
    await manager.close()


async def test_permanent_report_upsert_failure_fails_after_retries(tmp_path, monkeypatch) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await _completed_run(manager)

    calls = 0

    async def failing_upsert(rid: str, version: int, payload: dict, markdown: str) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"cannot read {tmp_path / 'data' / rid / 'report.json'}")

    monkeypatch.setattr(worker.reports, "upsert", failing_upsert)
    await worker.commands.enqueue(run_id, "start")

    assert await worker.tick() is True
    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "failed"
    assert commands[-1].error == "failed to record the run report"
    assert calls == 3
    await manager.close()


async def test_malformed_report_version_is_recorded_not_fatal(tmp_path) -> None:
    manager, worker = await _worker(tmp_path)
    run_id = await manager.create_run(paste_input(), budget())
    await manager.start_existing(run_id)
    await manager.approve(run_id, approved=True)

    paths = run_paths(worker.data_dir, run_id)
    payload = json.loads(paths.report_json.read_text(encoding="utf-8"))
    payload["version"] = "not-a-number"
    paths.report_json.write_text(json.dumps(payload), encoding="utf-8")

    await worker.commands.enqueue(run_id, "start")
    assert await worker.tick() is True

    snapshot = await worker.reports.get(run_id)
    assert snapshot is not None
    assert snapshot.version == 0
    assert snapshot.payload["version_error"]
    assert await worker.artifacts.get(run_id, "report.json") is not None
    commands = await worker.commands.list_for_run(run_id)
    assert commands[-1].status == "done"
    await manager.close()


async def _seed_upload(
    tmp_path: Path,
    factory,
    upload_id: str,
    *,
    manifest_sha256: str | None = None,
) -> None:
    paths = upload_paths(tmp_path / "data", upload_id)
    paths.root.mkdir(parents=True)
    with zipfile.ZipFile(paths.package, "w") as archive:
        archive.writestr("a.txt", b"hello")
    manifest = extract_archive(paths.package, paths.extracted)
    if manifest_sha256 is not None:
        manifest["files"][0]["sha256"] = manifest_sha256
    repo = UploadRepository(factory)
    await repo.create(
        upload_id,
        filename="p.zip",
        size_bytes=paths.package.stat().st_size,
        sha256=hashlib.sha256(paths.package.read_bytes()).hexdigest(),
        format="zip",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    await repo.mark_extracted(upload_id, manifest)


async def _run_upload(manager: RunManager, upload_id: str) -> str:
    run_id = await manager.create_run(
        RunInput(kind=InputKind.UPLOAD, upload_id=upload_id),
        budget(),
    )
    await RunCommandRepository(manager.session_factory).enqueue(run_id, "start")
    return run_id


async def test_worker_copies_upload_evidence_before_start(tmp_path: Path) -> None:
    config = fake_config(tmp_path)
    manager = RunManager(config, dependencies=dependencies(tmp_path))
    await manager.setup()
    try:
        upload_id = "upload_test1"
        await _seed_upload(tmp_path, manager.session_factory, upload_id)
        run_id = await _run_upload(manager, upload_id)
        worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir, poll_interval_s=0.01)

        assert await worker.tick() is True

        assert (run_paths(config.storage.data_dir, run_id).evidence / "a.txt").read_bytes() == b"hello"
        row = await worker.runs.get(run_id)
        assert row is not None and row.status == "waiting_approval"
    finally:
        await manager.close()


async def test_worker_rejects_unsafe_manifest_paths(tmp_path: Path) -> None:
    config = fake_config(tmp_path)
    manager = RunManager(config, dependencies=dependencies(tmp_path))
    await manager.setup()
    try:
        upload_id = "upload_unsafe"
        await _seed_upload(tmp_path, manager.session_factory, upload_id)
        repo = UploadRepository(manager.session_factory)
        worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir, poll_interval_s=0.01)
        digest = hashlib.sha256(b"hello").hexdigest()
        for unsafe in ("../escape.txt", "/abs.txt", "..\\escape.txt", "C:/escape.txt"):
            await repo.mark_extracted(
                upload_id,
                {"format": "zip", "files": [{"path": unsafe, "size": 5, "sha256": digest}]},
            )
            run_id = await _run_upload(manager, upload_id)

            assert await worker.tick() is True

            commands = await worker.commands.list_for_run(run_id)
            assert commands[-1].status == "failed"
            assert "unsafe" in (commands[-1].error or "")
            assert not run_paths(config.storage.data_dir, run_id).evidence.exists()
    finally:
        await manager.close()


async def test_worker_fails_the_command_on_a_checksum_mismatch(tmp_path: Path) -> None:
    config = fake_config(tmp_path)
    manager = RunManager(config, dependencies=dependencies(tmp_path))
    await manager.setup()
    try:
        upload_id = "upload_badsum"
        await _seed_upload(tmp_path, manager.session_factory, upload_id, manifest_sha256="0" * 64)
        run_id = await _run_upload(manager, upload_id)
        worker = RunWorker.from_manager(manager, data_dir=config.storage.data_dir, poll_interval_s=0.01)

        assert await worker.tick() is True

        commands = await worker.commands.list_for_run(run_id)
        assert commands[-1].status == "failed"
        assert "checksum" in (commands[-1].error or "")
        assert not (run_paths(config.storage.data_dir, run_id).evidence / "a.txt").exists()
    finally:
        await manager.close()
