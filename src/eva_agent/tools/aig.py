"""Run one authorized AIG model-redteam scan and persist its raw result."""

from __future__ import annotations

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
        paths = run_paths(data_dir, payload.run_id)
        paths.aig_dir.mkdir(parents=True, exist_ok=True)
        audit_error = False

        async def record(event: AuditEvent) -> None:
            nonlocal audit_error
            try:
                await audit.record(event)
            except Exception:
                # Audit must never fail the scan, but the caller needs to know.
                audit_error = True

        try:
            eval_token = resolve_api_key(eval_model, env)
        except MissingSecretError:
            await record(
                AuditEvent(
                    run_id=payload.run_id,
                    actor="cli:admin",
                    action="active_scan.failed",
                    outcome="failed",
                    target=payload.target.name,
                    authorization_ref=payload.authorization_ref,
                    scan_kind=SCAN_KIND,
                    detail={"error_type": "missing_eval_model_key"},
                )
            )
            raise
        target_token = env.get(payload.target.api_key_env or "", "")

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
            await record(
                AuditEvent(
                    run_id=payload.run_id,
                    actor="cli:admin",
                    action="active_scan.failed",
                    outcome="failed",
                    target=payload.target.name,
                    authorization_ref=payload.authorization_ref,
                    scan_kind=SCAN_KIND,
                    detail={"error_type": type(exc).__name__},
                )
            )
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
            await record(
                AuditEvent(
                    run_id=payload.run_id,
                    actor="cli:admin",
                    action="active_scan.failed",
                    outcome="failed",
                    target=payload.target.name,
                    authorization_ref=payload.authorization_ref,
                    scan_kind=SCAN_KIND,
                    detail={"session_id": session_id, "error_type": type(exc).__name__},
                )
            )
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
            await record(
                AuditEvent(
                    run_id=payload.run_id,
                    actor="cli:admin",
                    action="active_scan.failed",
                    outcome="failed",
                    target=payload.target.name,
                    authorization_ref=payload.authorization_ref,
                    scan_kind=SCAN_KIND,
                    detail={"session_id": session_id, "error_type": type(exc).__name__},
                )
            )
            raise

        try:
            data = await client.result(session_id)
            reports = parse_result(data)
        except AigError as exc:
            await record(
                AuditEvent(
                    run_id=payload.run_id,
                    actor="cli:admin",
                    action="active_scan.failed",
                    outcome="failed",
                    target=payload.target.name,
                    authorization_ref=payload.authorization_ref,
                    scan_kind=SCAN_KIND,
                    detail={"session_id": session_id, "error_type": type(exc).__name__},
                )
            )
            raise
        raw_path = paths.aig_dir / f"{session_id}.json"
        raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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

    return ToolSpec(
        name="run_aig_scan",
        description="Run one authorized AIG model-redteam scan.",
        input_model=AigScanInput,
        output_model=AigScanOutput,
        permission="admin",
        timeout_s=660.0,
        side_effects=True,
        idempotent=False,
        handler=handler,
    )
