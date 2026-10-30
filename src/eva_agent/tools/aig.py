"""Run one authorized AIG model-redteam scan and persist its raw result."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

from pydantic import BaseModel, Field

from eva_agent.adapters.aig.body import build_create_body
from eva_agent.adapters.aig.client import AigScanClient
from eva_agent.adapters.aig.errors import AigError, AigTimeout
from eva_agent.adapters.aig.models import AigReport, parse_result
from eva_agent.config.settings import AigSettings, MissingSecretError, ModelConfig, resolve_api_key
from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.target import TargetSpec
from eva_agent.runtime.artifacts import run_paths
from eva_agent.security.audit import AuditSink
from eva_agent.tools.registry import ToolSpec

SCAN_KIND = "model_redteam_report"
# Extra wall-clock room on top of three HTTP timeouts (create, wait, result) so
# the registry deadline never undercuts the configured scan wait window.
AIG_TOOL_TIMEOUT_MARGIN_S = 30.0
# AIG's CLI refuses an empty model token. A target without api_key_env accepts
# anonymous access (our probe sends no Authorization header for it, see
# adapters/target/secrets.py), so send this placeholder instead of "".
ANONYMOUS_TARGET_TOKEN = "not-required"
# Marker attached to a re-raised exception once a terminal audit row exists, so
# callers (the security expert) do not write a duplicate active_scan.failed row.
AUDIT_RECORDED_ATTR = "audit_recorded"


def _mark_audit_recorded(exc: BaseException) -> None:
    setattr(exc, AUDIT_RECORDED_ATTR, True)


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
        audit_error = False
        terminal_audit_written = False
        session_id: str | None = None

        async def record(event: AuditEvent) -> None:
            nonlocal audit_error
            try:
                await audit.record(event)
            except Exception:
                # Audit must never fail the scan, but the caller needs to know.
                audit_error = True

        async def record_failed(error_type: str) -> None:
            nonlocal terminal_audit_written
            if terminal_audit_written:
                return
            terminal_audit_written = True
            detail: dict[str, str] = {"error_type": error_type}
            if session_id is not None:
                detail["session_id"] = session_id
            await record(
                AuditEvent(
                    run_id=payload.run_id,
                    actor="cli:admin",
                    action="active_scan.failed",
                    outcome="failed",
                    target=payload.target.name,
                    authorization_ref=payload.authorization_ref,
                    scan_kind=SCAN_KIND,
                    detail=detail,
                )
            )

        try:
            paths = run_paths(data_dir, payload.run_id)
            paths.aig_dir.mkdir(parents=True, exist_ok=True)
            try:
                eval_token = resolve_api_key(eval_model, env)
            except MissingSecretError:
                await record_failed("missing_eval_model_key")
                raise
            if payload.target.api_key_env:
                target_token = env.get(payload.target.api_key_env)
                if not target_token:
                    await record_failed("missing_target_key")
                    raise MissingSecretError(f"environment variable {payload.target.api_key_env} is not set")
            else:
                target_token = ANONYMOUS_TARGET_TOKEN

            try:
                session_id = await client.create(
                    build_create_body(
                        target=payload.target,
                        scan_base_url=payload.scan_base_url,
                        eval_model=eval_model,
                        eval_token=eval_token,
                        target_token=target_token,
                        scan=settings.scan,
                    )
                )
            except AigError as exc:
                await record_failed(type(exc).__name__)
                raise
            await record(
                AuditEvent(
                    run_id=payload.run_id,
                    actor="cli:admin",
                    action="active_scan.started",
                    outcome="started",
                    target=payload.target.name,
                    authorization_ref=payload.authorization_ref,
                    scan_kind=SCAN_KIND,
                    detail={"session_id": session_id},
                )
            )
            try:
                await client.wait(session_id, max_wait_s=payload.max_wait_s)
            except AigTimeout as exc:
                await record_failed(type(exc).__name__)
                await record(
                    AuditEvent(
                        run_id=payload.run_id,
                        actor="cli:admin",
                        action="scan.cancelled",
                        outcome="cancelled",
                        target=payload.target.name,
                        authorization_ref=payload.authorization_ref,
                        scan_kind=SCAN_KIND,
                        detail={"session_id": session_id, "reason": "timeout"},
                    )
                )
                raise
            except AigError as exc:
                await record_failed(type(exc).__name__)
                raise

            try:
                data = await client.result(session_id)
                reports = parse_result(data)
            except AigError as exc:
                await record_failed(type(exc).__name__)
                raise
            raw_path = paths.aig_dir / f"{session_id}.json"
            try:
                raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError:
                await record_failed("OSError")
                raise
            total = sum(report.total for report in reports)
            jailbreak = sum(report.jailbreak for report in reports)
            findings = sum(
                1
                for report in reports
                for item in report.extra_body.vulnerability_results
                if (item.asr if item.asr is not None else (item.jailbreak / item.total if item.total else 0)) > 0
            )
            await record(
                AuditEvent(
                    run_id=payload.run_id,
                    actor="cli:admin",
                    action="active_scan.completed",
                    outcome="completed",
                    target=payload.target.name,
                    authorization_ref=payload.authorization_ref,
                    scan_kind=SCAN_KIND,
                    detail={
                        "session_id": session_id,
                        "total": total,
                        "jailbreak": jailbreak,
                        "asr": round(jailbreak / total, 4) if total else None,
                        "findings": findings,
                    },
                )
            )
            return AigScanOutput(
                session_id=session_id,
                reports=reports,
                datasets=list(settings.scan.datasets),
                num_prompts=settings.scan.num_prompts,
                raw_result_path=str(raw_path),
                audit_error=audit_error,
            )
        except asyncio.CancelledError as exc:
            # The registry cancels the handler when its own timeout fires.
            await record_failed("TimeoutError")
            _mark_audit_recorded(exc)
            raise
        except TimeoutError as exc:
            # asyncio.TimeoutError is an alias of the builtin on supported Pythons.
            await record_failed("TimeoutError")
            _mark_audit_recorded(exc)
            raise
        except Exception as exc:
            # Any other failure (for example the pre-scan mkdir) still leaves a
            # terminal audit row; record_failed dedups against earlier branches.
            await record_failed(type(exc).__name__)
            _mark_audit_recorded(exc)
            raise

    return ToolSpec(
        name="run_aig_scan",
        description="Run one authorized AIG model-redteam scan.",
        input_model=AigScanInput,
        output_model=AigScanOutput,
        permission="admin",
        timeout_s=settings.max_wait_s + 3 * settings.timeout_s + AIG_TOOL_TIMEOUT_MARGIN_S,
        side_effects=True,
        idempotent=False,
        handler=handler,
    )
