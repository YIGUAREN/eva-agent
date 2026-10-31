"""Run one authorized AIG scan and persist its raw result."""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eva_agent.adapters.aig.attachments import attachment_names
from eva_agent.adapters.aig.body import (
    build_agent_scan_body,
    build_create_body,
    build_skill_scan_body,
)
from eva_agent.adapters.aig.client import AigScanClient
from eva_agent.adapters.aig.errors import AigError, AigTimeout
from eva_agent.adapters.aig.models import (
    AigAgentReport,
    AigReport,
    AigSkillReport,
    parse_agent_result,
    parse_result,
    parse_skill_result,
)
from eva_agent.config.settings import AigSettings, MissingSecretError, ModelConfig, resolve_api_key
from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.target import TargetSpec
from eva_agent.runtime.artifacts import RunPaths, run_paths
from eva_agent.security.audit import AuditSink
from eva_agent.tools.evidence import EvidenceAccessError, resolve_evidence_path
from eva_agent.tools.registry import ToolSpec

SCAN_KIND = "model_redteam_report"
AGENT_SCAN_KIND = "agent_scan"
SKILL_SCAN_KIND = "skill_scan"
# Extra wall-clock room on top of a tool's HTTP timeouts (create, wait, result,
# plus upload for the skill tool and download for the model-redteam tool) so the
# registry deadline never undercuts the configured scan wait window.
AIG_TOOL_TIMEOUT_MARGIN_S = 30.0
# AIG's CLI refuses an empty model token. A target without api_key_env accepts
# anonymous access (our probe sends no Authorization header for it, see
# adapters/target/secrets.py), so send this placeholder instead of "".
ANONYMOUS_TARGET_TOKEN = "not-required"
# Marker attached to a re-raised exception once a terminal audit row exists, so
# callers (the security expert) do not write a duplicate active_scan.failed row.
AUDIT_RECORDED_ATTR = "audit_recorded"


class AigPackageError(RuntimeError):
    """The evidence package cannot be uploaded as a single archive."""


def _mark_audit_recorded(exc: BaseException) -> None:
    setattr(exc, AUDIT_RECORDED_ATTR, True)


class _Auditor:
    """Record the started/completed/failed audit trail for one scan."""

    def __init__(
        self,
        *,
        audit: AuditSink,
        run_id: str,
        target: str,
        authorization_ref: str,
        scan_kind: str,
    ) -> None:
        self._audit = audit
        self._run_id = run_id
        self._target = target
        self._authorization_ref = authorization_ref
        self._scan_kind = scan_kind
        self.error = False
        self.terminal = False
        self.session_id: str | None = None

    async def record(self, event: AuditEvent) -> None:
        try:
            await self._audit.record(event)
        except Exception:
            # Audit must never fail the scan, but the caller needs to know.
            self.error = True

    def _event(self, action: str, outcome: str, detail: dict[str, Any]) -> AuditEvent:
        return AuditEvent(
            run_id=self._run_id,
            actor="cli:admin",
            action=action,
            outcome=outcome,
            target=self._target,
            authorization_ref=self._authorization_ref,
            scan_kind=self._scan_kind,
            detail=detail,
        )

    async def failed(self, error_type: str) -> None:
        if self.terminal:
            return
        self.terminal = True
        detail: dict[str, Any] = {"error_type": error_type}
        if self.session_id is not None:
            detail["session_id"] = self.session_id
        await self.record(self._event("active_scan.failed", "failed", detail))

    async def started(self) -> None:
        detail = {"session_id": self.session_id or ""}
        await self.record(self._event("active_scan.started", "started", detail))

    async def cancelled(self, reason: str) -> None:
        detail = {"session_id": self.session_id or "", "reason": reason}
        await self.record(self._event("scan.cancelled", "cancelled", detail))

    async def completed(self, detail: dict[str, Any]) -> None:
        self.terminal = True
        await self.record(self._event("active_scan.completed", "completed", detail))

    async def attachment_failed(self, reason: str) -> None:
        # Attachment metadata is sensitive: record only a static reason.
        await self.record(self._event("active_scan.attachment_failed", "failed", {"reason": reason}))


class AigScanInput(BaseModel):
    run_id: str = Field(min_length=1)
    target: TargetSpec
    scan_base_url: str = Field(min_length=1)
    authorization_ref: str = Field(min_length=1)
    max_wait_s: float = Field(gt=0)


class AigScanOutput(BaseModel):
    session_id: str = Field(min_length=1)
    reports: list[AigReport]
    datasets: list[str]
    num_prompts: int = Field(ge=0)
    raw_result_path: str = Field(min_length=1)
    audit_error: bool = False


class AigAgentScanInput(BaseModel):
    run_id: str = Field(min_length=1)
    target: TargetSpec
    authorization_ref: str = Field(min_length=1)
    max_wait_s: float = Field(gt=0)


class AigAgentScanOutput(BaseModel):
    session_id: str = Field(min_length=1)
    report: AigAgentReport
    raw_result_path: str = Field(min_length=1)
    audit_error: bool = False


class AigSkillScanInput(BaseModel):
    run_id: str = Field(min_length=1)
    root: str = Field(min_length=1)
    files: list[str] = Field(min_length=1)
    authorization_ref: str = Field(min_length=1)
    max_wait_s: float = Field(gt=0)


class AigSkillScanOutput(BaseModel):
    session_id: str = Field(min_length=1)
    report: AigSkillReport
    raw_result_path: str = Field(min_length=1)
    audit_error: bool = False


async def _run_lifecycle(
    *,
    auditor: _Auditor,
    client: AigScanClient,
    body: dict[str, Any],
    parse: Callable[[dict[str, Any]], Any],
    aig_dir: Path,
    max_wait_s: float,
) -> tuple[str, Any, Path]:
    try:
        session_id = await client.create(body)
    except AigError as exc:
        await auditor.failed(type(exc).__name__)
        raise
    auditor.session_id = session_id
    await auditor.started()
    try:
        await client.wait(session_id, max_wait_s=max_wait_s)
    except AigTimeout as exc:
        await auditor.failed(type(exc).__name__)
        await auditor.cancelled("timeout")
        raise
    except AigError as exc:
        await auditor.failed(type(exc).__name__)
        raise
    try:
        data = await client.result(session_id)
        parsed = parse(data)
    except AigError as exc:
        await auditor.failed(type(exc).__name__)
        raise
    raw_path = aig_dir / f"{session_id}.json"
    try:
        raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        await auditor.failed("OSError")
        raise
    return session_id, parsed, raw_path


async def _save_attachment(
    *,
    auditor: _Auditor,
    client: AigScanClient,
    paths: RunPaths,
    raw_path: Path,
) -> None:
    """Fetch AIG's model-redteam CSVs; a failure never fails the scan."""
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        await auditor.attachment_failed("unreadable_result")
        return
    names = attachment_names(payload if isinstance(payload, dict) else {})
    if not names:
        return
    if not callable(getattr(client, "download", None)):
        # A structurally compatible client that predates the download contract must
        # degrade (and be audited) instead of raising AttributeError after the scan
        # already succeeded.
        await auditor.attachment_failed("download_unsupported")
        return
    session_id = raw_path.stem
    # AIG can report several attachments; download each distinct name once. The
    # first keeps the acceptance-tested "<session_id>.csv" name, later ones are
    # numbered from 2 so Task 7's aig/*.csv glob picks them up too.
    distinct = list(dict.fromkeys(names))
    for index, name in enumerate(distinct, start=1):
        filename = f"{session_id}.csv" if index == 1 else f"{session_id}-{index}.csv"
        try:
            content = await client.download(session_id, name)
            (paths.aig_dir / filename).write_bytes(content)
        except (AigError, OSError) as exc:
            await auditor.attachment_failed(type(exc).__name__)


def _package_archive(root: str, files: list[str], *, limit: int) -> bytes:
    """Zip the listed evidence files; refuse anything that escapes the root or the size cap."""
    buffer = io.BytesIO()
    total = 0
    root_path = Path(root).resolve()
    try:
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in files:
                path = resolve_evidence_path(root, relative)
                # Name members by their resolved path inside the root so a caller
                # cannot smuggle ".." segments or a drive prefix into the archive.
                try:
                    member = path.relative_to(root_path).as_posix()
                except ValueError:
                    raise AigPackageError("skill package member escapes the evidence root") from None
                try:
                    data = path.read_bytes()
                except OSError:
                    raise AigPackageError("skill package member is not readable") from None
                total += len(data)
                if total > limit:
                    raise AigPackageError(f"skill package exceeds {limit} bytes")
                archive.writestr(member, data)
    except EvidenceAccessError as exc:
        raise AigPackageError(f"skill package member is not readable: {exc.reason}") from None
    return buffer.getvalue()


def build_run_aig_scan_tool(
    *,
    client: AigScanClient,
    settings: AigSettings,
    eval_model: ModelConfig,
    env: Mapping[str, str],
    audit: AuditSink,
    data_dir: str,
) -> ToolSpec:
    async def handler(payload: AigScanInput) -> AigScanOutput:
        auditor = _Auditor(
            audit=audit,
            run_id=payload.run_id,
            target=payload.target.name,
            authorization_ref=payload.authorization_ref,
            scan_kind=SCAN_KIND,
        )
        try:
            paths = run_paths(data_dir, payload.run_id)
            paths.aig_dir.mkdir(parents=True, exist_ok=True)
            try:
                eval_token = resolve_api_key(eval_model, env)
            except MissingSecretError:
                await auditor.failed("missing_eval_model_key")
                raise
            if payload.target.api_key_env:
                target_token = env.get(payload.target.api_key_env)
                if not target_token:
                    await auditor.failed("missing_target_key")
                    raise MissingSecretError(f"environment variable {payload.target.api_key_env} is not set")
            else:
                target_token = ANONYMOUS_TARGET_TOKEN
            body = build_create_body(
                target=payload.target,
                scan_base_url=payload.scan_base_url,
                eval_model=eval_model,
                eval_token=eval_token,
                target_token=target_token,
                scan=settings.scan,
            )
            session_id, reports, raw_path = await _run_lifecycle(
                auditor=auditor,
                client=client,
                body=body,
                parse=parse_result,
                aig_dir=paths.aig_dir,
                max_wait_s=payload.max_wait_s,
            )
            await _save_attachment(
                auditor=auditor,
                client=client,
                paths=paths,
                raw_path=raw_path,
            )
            total = sum(report.total for report in reports)
            jailbreak = sum(report.jailbreak for report in reports)
            findings = sum(
                1
                for report in reports
                for item in report.extra_body.vulnerability_results
                if (item.asr if item.asr is not None else (item.jailbreak / item.total if item.total else 0)) > 0
            )
            await auditor.completed(
                {
                    "session_id": session_id,
                    "total": total,
                    "jailbreak": jailbreak,
                    "asr": round(jailbreak / total, 4) if total else None,
                    "findings": findings,
                }
            )
            return AigScanOutput(
                session_id=session_id,
                reports=reports,
                datasets=list(settings.scan.datasets),
                num_prompts=settings.scan.num_prompts,
                raw_result_path=str(raw_path),
                audit_error=auditor.error,
            )
        except asyncio.CancelledError as exc:
            # The registry cancels the handler when its own timeout fires.
            await auditor.failed("TimeoutError")
            _mark_audit_recorded(exc)
            raise
        except TimeoutError as exc:
            # asyncio.TimeoutError is an alias of the builtin on supported Pythons.
            await auditor.failed("TimeoutError")
            _mark_audit_recorded(exc)
            raise
        except Exception as exc:
            # Any other failure (for example the pre-scan mkdir) still leaves a
            # terminal audit row; auditor.failed dedups against earlier branches.
            await auditor.failed(type(exc).__name__)
            _mark_audit_recorded(exc)
            raise

    return ToolSpec(
        name="run_aig_scan",
        description="Run one authorized AIG model-redteam scan.",
        input_model=AigScanInput,
        output_model=AigScanOutput,
        permission="admin",
        # Create + last wait poll + result + download consume one timeout each.
        timeout_s=settings.max_wait_s + 4 * settings.timeout_s + AIG_TOOL_TIMEOUT_MARGIN_S,
        side_effects=True,
        idempotent=False,
        handler=handler,
    )


def build_run_aig_agent_scan_tool(
    *,
    client: AigScanClient,
    settings: AigSettings,
    eval_model: ModelConfig,
    env: Mapping[str, str],
    audit: AuditSink,
    data_dir: str,
) -> ToolSpec:
    async def handler(payload: AigAgentScanInput) -> AigAgentScanOutput:
        auditor = _Auditor(
            audit=audit,
            run_id=payload.run_id,
            target=payload.target.name,
            authorization_ref=payload.authorization_ref,
            scan_kind=AGENT_SCAN_KIND,
        )
        try:
            paths = run_paths(data_dir, payload.run_id)
            paths.aig_dir.mkdir(parents=True, exist_ok=True)
            try:
                eval_token = resolve_api_key(eval_model, env)
            except MissingSecretError:
                await auditor.failed("missing_eval_model_key")
                raise
            headers = dict(payload.target.headers)
            for header, env_name in payload.target.header_env.items():
                value = env.get(env_name)
                if not value:
                    await auditor.failed("missing_target_key")
                    raise MissingSecretError(f"environment variable {env_name} is not set")
                headers[header] = value
            body = build_agent_scan_body(
                target=payload.target,
                headers=headers,
                eval_model=eval_model,
                eval_token=eval_token,
                scan=settings.agent_scan,
            )
            session_id, report, raw_path = await _run_lifecycle(
                auditor=auditor,
                client=client,
                body=body,
                parse=parse_agent_result,
                aig_dir=paths.aig_dir,
                max_wait_s=payload.max_wait_s,
            )
            await auditor.completed(
                {
                    "session_id": session_id,
                    "total": str(report.total_tests),
                    "findings": str(len(report.results)),
                    "risk_type": report.risk_type,
                }
            )
            return AigAgentScanOutput(
                session_id=session_id,
                report=report,
                raw_result_path=str(raw_path),
                audit_error=auditor.error,
            )
        except asyncio.CancelledError as exc:
            await auditor.failed("TimeoutError")
            _mark_audit_recorded(exc)
            raise
        except TimeoutError as exc:
            await auditor.failed("TimeoutError")
            _mark_audit_recorded(exc)
            raise
        except Exception as exc:
            await auditor.failed(type(exc).__name__)
            _mark_audit_recorded(exc)
            raise

    return ToolSpec(
        name="run_aig_agent_scan",
        description="Run one authorized AIG agent security scan against an agent_http target.",
        input_model=AigAgentScanInput,
        output_model=AigAgentScanOutput,
        permission="admin",
        timeout_s=settings.max_wait_s + 3 * settings.timeout_s + AIG_TOOL_TIMEOUT_MARGIN_S,
        side_effects=True,
        idempotent=False,
        handler=handler,
    )


def build_run_aig_skill_scan_tool(
    *,
    client: AigScanClient,
    settings: AigSettings,
    eval_model: ModelConfig,
    env: Mapping[str, str],
    audit: AuditSink,
    data_dir: str,
) -> ToolSpec:
    async def handler(payload: AigSkillScanInput) -> AigSkillScanOutput:
        auditor = _Auditor(
            audit=audit,
            run_id=payload.run_id,
            target="skill_package",
            authorization_ref=payload.authorization_ref,
            scan_kind=SKILL_SCAN_KIND,
        )
        try:
            paths = run_paths(data_dir, payload.run_id)
            paths.aig_dir.mkdir(parents=True, exist_ok=True)
            try:
                eval_token = resolve_api_key(eval_model, env)
            except MissingSecretError:
                await auditor.failed("missing_eval_model_key")
                raise
            archive_bytes = _package_archive(
                payload.root,
                payload.files,
                limit=settings.skill_scan.max_package_bytes,
            )
            file_url = await client.upload(f"{payload.run_id}-skill.zip", archive_bytes)
            body = build_skill_scan_body(
                eval_model=eval_model,
                eval_token=eval_token,
                file_url=file_url,
                scan=settings.skill_scan,
            )
            session_id, report, raw_path = await _run_lifecycle(
                auditor=auditor,
                client=client,
                body=body,
                parse=parse_skill_result,
                aig_dir=paths.aig_dir,
                max_wait_s=payload.max_wait_s,
            )
            await auditor.completed(
                {
                    "session_id": session_id,
                    "findings": str(len(report.results)),
                    "score": str(report.score),
                }
            )
            return AigSkillScanOutput(
                session_id=session_id,
                report=report,
                raw_result_path=str(raw_path),
                audit_error=auditor.error,
            )
        except asyncio.CancelledError as exc:
            await auditor.failed("TimeoutError")
            _mark_audit_recorded(exc)
            raise
        except TimeoutError as exc:
            await auditor.failed("TimeoutError")
            _mark_audit_recorded(exc)
            raise
        except Exception as exc:
            await auditor.failed(type(exc).__name__)
            _mark_audit_recorded(exc)
            raise

    return ToolSpec(
        name="run_aig_skill_scan",
        description="Upload one skill package and run one authorized AIG skill scan.",
        input_model=AigSkillScanInput,
        output_model=AigSkillScanOutput,
        permission="admin",
        # Upload + create + last wait poll + result all consume one timeout each.
        timeout_s=settings.max_wait_s + 4 * settings.timeout_s + AIG_TOOL_TIMEOUT_MARGIN_S,
        side_effects=True,
        idempotent=False,
        handler=handler,
    )
