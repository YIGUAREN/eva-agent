import asyncio
import json

from eva_agent.domain.run import RunStatus
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.manager import RunManager
from eva_agent.worker.runner import RunWorker
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
    assert await worker.artifacts.get(run_id, "report.json") is not None
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
