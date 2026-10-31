"""Claim run commands and drive the RunManager."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from uuid import uuid4

from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind, RunInput, RunStatus
from eva_agent.runtime.artifacts import RunPaths, run_paths
from eva_agent.runtime.manager import RunManager, RunStateError
from eva_agent.runtime.uploads import upload_paths, validate_upload_id
from eva_agent.storage.commands import RunCommandRepository
from eva_agent.storage.models import RunCommandRow
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
from eva_agent.storage.repository import RunRepository
from eva_agent.storage.uploads import UPLOAD_EXTRACTED, UploadRepository

logger = logging.getLogger(__name__)

HANDLED_KINDS = frozenset({"start", "approve", "cancel", "resume", "budget_decision"})

TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})

REPORT_RECORD_ERROR = "failed to record the run report"
GENERIC_COMMAND_ERROR = "command failed"


class ReportRecordError(RuntimeError):
    """Raised when a completed run's report or artifacts cannot be persisted."""


def _public_error(exc: Exception) -> str:
    """Return error text that is safe to persist.

    Only locally constructed errors may expose their message; anything else can
    embed absolute paths or raw provider output, which must never reach storage.
    """
    if isinstance(exc, ReportRecordError):
        return REPORT_RECORD_ERROR
    # Only an exact ``ValueError`` is locally constructed. Subclasses such as
    # json.JSONDecodeError can carry raw document text and must stay hidden.
    if isinstance(exc, RunStateError) or type(exc) is ValueError:
        return str(exc)
    return GENERIC_COMMAND_ERROR


def _report_version(payload: dict, run_id: str) -> int:
    """Return a usable report version, recording a malformed one instead of raising."""
    raw = payload.get("version", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("report for %s has a malformed version %r; storing version 0", run_id, raw)
        payload["version"] = 0
        payload["version_error"] = "the report version was malformed and has been reset to 0"
        return 0


_AIG_FILE = re.compile(r"[A-Za-z0-9_-]{1,128}\.(json|csv)")


def _artifact_specs(paths: RunPaths) -> list[tuple[str, Path, str, str]]:
    specs = [
        ("report.json", paths.report_json, "admin", "application/json"),
        ("report.md", paths.report_markdown, "admin", "text/markdown; charset=utf-8"),
        ("plan.approved.yaml", paths.plan_approved, "admin", "application/yaml"),
        ("target_probe.jsonl", paths.target_probe, "admin", "application/x-ndjson"),
    ]
    for path in sorted(paths.aig_dir.glob("*")):
        if not path.is_file() or _AIG_FILE.fullmatch(path.name) is None:
            continue
        content_type = "application/json" if path.suffix == ".json" else "text/csv; charset=utf-8"
        specs.append((f"aig/{path.name}", path, "admin", content_type))
    return specs


class RunWorker:
    def __init__(
        self,
        *,
        manager: RunManager,
        commands: RunCommandRepository,
        runs: RunRepository,
        reports: ReportRepository,
        artifacts: ArtifactRepository,
        uploads: UploadRepository,
        data_dir: str,
        poll_interval_s: float = 0.5,
        stale_after_s: float = 600.0,
    ) -> None:
        self.manager = manager
        self.commands = commands
        self.runs = runs
        self.reports = reports
        self.artifacts = artifacts
        self.uploads = uploads
        self.data_dir = data_dir
        self.poll_interval_s = poll_interval_s
        self.stale_after_s = stale_after_s

    @classmethod
    def from_manager(
        cls,
        manager: RunManager,
        *,
        data_dir: str,
        poll_interval_s: float = 0.5,
    ) -> RunWorker:
        factory = manager.session_factory
        return cls(
            manager=manager,
            commands=RunCommandRepository(factory),
            runs=RunRepository(factory),
            reports=ReportRepository(factory),
            artifacts=ArtifactRepository(factory),
            uploads=UploadRepository(factory),
            data_dir=data_dir,
            poll_interval_s=poll_interval_s,
        )

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                worked = await self.tick()
            except Exception:
                # A transient storage failure must not terminate the worker: log it
                # and back off before retrying, just like an idle poll.
                logger.exception("worker tick failed; backing off")
                worked = False
            if not worked:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self.poll_interval_s)

    async def tick(self) -> bool:
        cutoff = datetime.now(UTC) - timedelta(seconds=self.stale_after_s)
        await self.commands.reset_stale(cutoff)
        command = await self.commands.claim_next()
        if command is None:
            return False
        try:
            await self._handle(command)
        except Exception as exc:
            logger.warning("command %s (%s) failed", command.id, command.kind, exc_info=True)
            # The underlying error may embed an absolute path, which the plan
            # forbids in error messages, so persist only sanitized text.
            await self.commands.finish(command.id, "failed", _public_error(exc)[:500])
        else:
            await self.commands.finish(command.id, "done")
        return True

    async def _handle(self, command: RunCommandRow) -> None:
        if command.kind not in HANDLED_KINDS:
            raise ValueError(f"unknown command kind: {command.kind}")
        row = await self.runs.get(command.run_id)
        if row is None:
            raise RunStateError(f"run '{command.run_id}' does not exist")
        status = RunStatus(row.status)

        if command.kind == "start":
            if status is RunStatus.QUEUED:
                run_input = RunInput.model_validate(row.input_json)
                if run_input.kind is InputKind.UPLOAD:
                    await self._copy_upload(command.run_id, run_input)
                await self.manager.start_existing(command.run_id)
            elif status is RunStatus.RUNNING:
                await self._resume(command.run_id)
        elif command.kind == "resume":
            if status is RunStatus.RUNNING:
                await self._resume(command.run_id)
            elif status not in TERMINAL_STATUSES:
                # Forwarding a resume blindly would either corrupt the lifecycle
                # or hide the caller's mistake; name the status so it can be fixed.
                raise RunStateError(f"run '{command.run_id}' cannot be resumed from status '{status.value}'")
            # A terminal run cannot be resumed, but the command may be a replay
            # after a crash following completion; fall through to _after so the
            # report and artifact rows are repaired.
        elif command.kind == "approve":
            if status is RunStatus.WAITING_APPROVAL:
                await self.manager.approve(
                    command.run_id,
                    approved=bool(command.payload_json.get("approved")),
                    plan=command.payload_json.get("plan"),
                )
        elif command.kind == "budget_decision":
            if status in (RunStatus.WAITING_BUDGET, RunStatus.RUNNING):
                # RUNNING covers a retried delivery whose earlier decision was
                # persisted but whose resume crashed; the manager recovers it.
                payload = command.payload_json
                raw_budget = payload.get("budget")
                await self.manager.decide_budget(
                    command.run_id,
                    action=str(payload.get("action")),
                    confirm_overrun=bool(payload.get("confirm_overrun")),
                    budget=Budget.model_validate(raw_budget) if raw_budget is not None else None,
                )
        elif command.kind == "cancel":
            await self.manager.cancel(command.run_id)

        # Run _after for every handled command, including no-ops. A worker crash
        # between the graph finishing and _after leaves the command claimed, so
        # reset_stale requeues it; replaying a duplicate start/approve must still
        # backfill the report and artifact rows. _after is idempotent, so a
        # transient storage error is retried before the command is failed.
        for attempt in range(3):
            try:
                await self._after(command.run_id)
            except Exception as exc:
                if attempt == 2:
                    raise ReportRecordError(REPORT_RECORD_ERROR) from exc
                await asyncio.sleep(0.05 if attempt == 0 else 0.1)
            else:
                break

    async def _copy_upload(self, run_id: str, run_input: RunInput) -> None:
        """Copy a sandbox-extracted upload into the run workspace, verifying each file."""
        upload_id = run_input.upload_id or ""
        try:
            validate_upload_id(upload_id)
        except ValueError:
            # Never echo an untrusted id: it may be an absolute path.
            raise RunStateError("upload id is invalid") from None
        row = await self.uploads.get(upload_id)
        if row is None or row.status != UPLOAD_EXTRACTED or row.manifest is None:
            raise RunStateError(f"upload '{upload_id}' is not ready")
        source_root = upload_paths(self.data_dir, upload_id).extracted.resolve()
        destination_root = run_paths(self.data_dir, run_id).evidence.resolve()
        for item in row.manifest.get("files", []):
            raw_path = str(item["path"])
            relative = PurePosixPath(raw_path)
            if (
                "\\" in raw_path
                or relative.is_absolute()
                or (relative.parts and len(relative.parts[0]) == 2 and relative.parts[0][1] == ":")
                or ".." in relative.parts
            ):
                raise RunStateError("upload manifest contains an unsafe path")
            source = source_root.joinpath(*relative.parts).resolve()
            if not source.is_relative_to(source_root) or not source.is_file():
                raise RunStateError("upload file is missing")
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RunStateError("upload file checksum mismatch")
            target = destination_root.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(destination_root):
                raise RunStateError("upload manifest contains an unsafe path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    async def _resume(self, run_id: str) -> None:
        """Resume a RUNNING run, failing it when it has no checkpoint to resume from."""
        try:
            if await self.manager.paused_at_budget_gate(run_id):
                # The process died between the gate checkpoint and _sync.  The
                # helper parks the row only while the gate is still pending; a
                # recorded decision keeps the row RUNNING so it stays replayable.
                await self.manager.converge_from_checkpoint(run_id)
                return
            # The checkpoint may already be at END while the row still says
            # RUNNING (the process died before _sync).  Converge instead of
            # re-invoking a graph that already finished.
            if await self.manager.converge_from_checkpoint(run_id):
                return
            await self.manager.resume(run_id)
        except RunStateError as exc:
            row = await self.runs.get(run_id)
            # A RUNNING row without a checkpoint cannot make progress: the process
            # died between claiming the run and writing its first checkpoint, so
            # fail it instead of leaving it stuck forever.
            if (
                row is not None
                and row.status == RunStatus.RUNNING.value
                and await self.runs.claim(run_id, from_statuses=(RunStatus.RUNNING,), to_status=RunStatus.FAILED)
            ):
                await self.runs.set_error(run_id, _public_error(exc))
            raise

    async def _after(self, run_id: str) -> None:
        row = await self.runs.get(run_id)
        if row is None or row.status != RunStatus.COMPLETED.value:
            return
        paths = run_paths(self.data_dir, run_id)
        if not paths.report_json.is_file():
            return
        payload = json.loads(paths.report_json.read_text(encoding="utf-8"))
        markdown = paths.report_markdown.read_text(encoding="utf-8") if paths.report_markdown.is_file() else ""
        await self.reports.upsert(run_id, _report_version(payload, run_id), payload, markdown)
        for name, path, visibility, content_type in _artifact_specs(paths):
            if not path.is_file() or await self.artifacts.get(run_id, name) is not None:
                continue
            data = path.read_bytes()
            await self.artifacts.add(
                artifact_id=f"art_{uuid4().hex}",
                run_id=run_id,
                name=name,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                content_type=content_type,
                visibility=visibility,
            )
