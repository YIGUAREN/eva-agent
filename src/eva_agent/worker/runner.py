"""Claim run commands and drive the RunManager."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from eva_agent.domain.run import RunStatus
from eva_agent.runtime.artifacts import RunPaths, run_paths
from eva_agent.runtime.manager import RunManager, RunStateError
from eva_agent.storage.commands import RunCommandRepository
from eva_agent.storage.models import RunCommandRow
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
from eva_agent.storage.repository import RunRepository

logger = logging.getLogger(__name__)

HANDLED_KINDS = frozenset({"start", "approve", "cancel", "resume"})


def _artifact_specs(paths: RunPaths) -> list[tuple[str, Path, str, str]]:
    return [
        ("report.json", paths.report_json, "public", "application/json"),
        ("report.md", paths.report_markdown, "public", "text/markdown; charset=utf-8"),
        ("plan.approved.yaml", paths.plan_approved, "admin", "application/yaml"),
        ("target_probe.jsonl", paths.target_probe, "admin", "application/x-ndjson"),
    ]


class RunWorker:
    def __init__(
        self,
        *,
        manager: RunManager,
        commands: RunCommandRepository,
        runs: RunRepository,
        reports: ReportRepository,
        artifacts: ArtifactRepository,
        data_dir: str,
        poll_interval_s: float = 0.5,
        stale_after_s: float = 600.0,
    ) -> None:
        self.manager = manager
        self.commands = commands
        self.runs = runs
        self.reports = reports
        self.artifacts = artifacts
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
            data_dir=data_dir,
            poll_interval_s=poll_interval_s,
        )

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            worked = await self.tick()
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
            await self.commands.finish(command.id, "failed", f"{type(exc).__name__}: {exc}"[:500])
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
                await self.manager.start_existing(command.run_id)
            elif status is RunStatus.RUNNING:
                await self._resume(command.run_id)
            else:
                return
        elif command.kind == "resume":
            await self._resume(command.run_id)
        elif command.kind == "approve":
            if status is not RunStatus.WAITING_APPROVAL:
                return
            await self.manager.approve(
                command.run_id,
                approved=bool(command.payload_json.get("approved")),
                plan=command.payload_json.get("plan"),
            )
        elif command.kind == "cancel":
            await self.manager.cancel(command.run_id)

        await self._after(command.run_id)

    async def _resume(self, run_id: str) -> None:
        """Resume a RUNNING run, failing it when it has no checkpoint to resume from."""
        try:
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
                await self.runs.set_error(run_id, f"{type(exc).__name__}: {exc}")
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
        await self.reports.upsert(run_id, int(payload.get("version", 0)), payload, markdown)
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
