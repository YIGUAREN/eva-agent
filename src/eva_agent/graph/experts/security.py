"""Security expert: passive checks plus an authorized AIG active scan."""

from __future__ import annotations

from typing import Any, Literal

from eva_agent.adapters.aig.errors import AigError, AigTimeout
from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.probe import ProbeSummary
from eva_agent.domain.result import DimensionResult, Finding, Usage
from eva_agent.domain.run import InputKind
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.experts.rows import prepare_rows_with_indexes
from eva_agent.security.active import ActiveSummary, summarize_active
from eva_agent.security.passive import passive_specs, summarize_passive
from eva_agent.tools.aig import AigScanInput
from eva_agent.tools.graders import RunGradersInput

ActiveStatus = Literal["not_applicable", "refused", "unavailable", "failed", "completed"]
SCAN_KIND = "model_redteam_report"


class SecurityExpert:
    def __init__(self, deps: GraphDependencies) -> None:
        self._deps = deps

    async def run(
        self,
        *,
        run_id: str,
        plan: DimensionPlan,
        evidence: EvidenceBundle,
        probe: ProbeSummary | None,
        budget: Budget,
    ) -> DimensionResult:
        if plan.dimension is not Dimension.SECURITY:
            raise ValueError("security expert received a non-security plan")
        limit = plan.sample_size or self._deps.default_sample_size
        rows, source_indexes = await prepare_rows_with_indexes(self._deps.tools, evidence, run_id=run_id, limit=limit)
        if not rows:
            return self._failed("security 没有可评估的成功响应")

        specs = passive_specs(rows)
        try:
            output = await self._deps.tools.call("run_graders", RunGradersInput(specs=specs, rows=rows, run_id=run_id))
        except Exception as exc:
            return self._failed(f"security 被动检查失败：{type(exc).__name__}")

        passive = summarize_passive(output, specs=specs, rows=rows, evidence=evidence, source_indexes=source_indexes)
        await self._event(
            run_id,
            "security.passive.completed",
            {
                "rows": len(rows),
                "graders": [spec.name for spec in specs],
                "errors": len(passive.metrics.get("errors") or []),
            },
        )
        active = await self._active(run_id, evidence, probe, budget)
        return self._result(passive, active, len(specs))

    async def _audit(self, event: AuditEvent) -> bool:
        try:
            await self._deps.audit.record(event)
            return False
        except Exception:
            return True

    async def _event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        try:
            await self._deps.event_sink(run_id, event_type, payload)
        except Exception:
            return

    @staticmethod
    def _active_metrics(status: ActiveStatus, score: float | None) -> dict[str, Any]:
        return {
            "status": status,
            "session_id": None,
            "dataset": [],
            "num_prompts": 0,
            "asr": None,
            "jailbreak": 0,
            "total": 0,
            "vulnerabilities": [],
            "severity_counts": {},
            "raw_result_path": None,
            "degraded": status in {"failed", "unavailable"},
            "refused_reason": None,
            "score": score,
        }

    async def _active(
        self,
        run_id: str,
        evidence: EvidenceBundle,
        probe: ProbeSummary | None,
        budget: Budget,
    ) -> tuple[ActiveStatus, ActiveSummary | None, dict[str, Any], bool]:
        if evidence.kind is not InputKind.TARGET:
            return "not_applicable", None, self._active_metrics("not_applicable", None), False

        target = evidence.target
        if target is None:
            return "failed", None, self._active_metrics("failed", None), False
        if not self._deps.aig_configured or self._deps.scan_policy is None:
            audit_error = await self._audit(
                AuditEvent(
                    run_id=run_id,
                    actor="cli:admin",
                    action="active_scan.failed",
                    outcome="failed",
                    target=target.name,
                    scan_kind=SCAN_KIND,
                    detail={"error_type": "not_configured"},
                )
            )
            await self._event(
                run_id,
                "security.active.failed",
                {"target": target.name, "session_id": None, "error_type": "not_configured"},
            )
            return "unavailable", None, self._active_metrics("unavailable", None), audit_error

        probe_count = probe.total if probe is not None else 0
        decision = self._deps.scan_policy.decide(
            target=target,
            input_kind=evidence.kind,
            probe_count=probe_count,
            budget=budget,
        )
        audit_error = await self._audit(
            AuditEvent(
                run_id=run_id,
                actor="cli:admin",
                action="active_scan.requested",
                outcome="allowed" if decision.allowed else "refused",
                target=target.name,
                authorization_ref=decision.authorization_ref,
                scan_kind=SCAN_KIND,
                detail={
                    "reason": decision.reason,
                    "probe_count": probe_count,
                    "scan_requests": decision.scan_requests,
                    "scan_base_url": decision.scan_base_url,
                },
            )
        )
        await self._event(
            run_id,
            "security.active.requested",
            {"target": target.name, "allowed": decision.allowed, "reason": decision.reason},
        )
        if not decision.allowed:
            metrics = self._active_metrics("refused", None)
            metrics["refused_reason"] = decision.reason
            return "refused", None, metrics, audit_error

        try:
            output = await self._deps.tools.call(
                "run_aig_scan",
                AigScanInput(
                    run_id=run_id,
                    target=target,
                    scan_base_url=decision.scan_base_url or "",
                    authorization_ref=decision.authorization_ref or "",
                    max_wait_s=decision.max_wait_s or 0.0,
                ),
            )
        except AigTimeout:
            await self._event(
                run_id,
                "security.active.failed",
                {"target": target.name, "session_id": None, "error_type": "AigTimeout"},
            )
            return "failed", None, self._active_metrics("failed", None), audit_error
        except AigError as exc:
            await self._event(
                run_id,
                "security.active.failed",
                {"target": target.name, "session_id": None, "error_type": type(exc).__name__},
            )
            return "failed", None, self._active_metrics("failed", None), audit_error
        except Exception as exc:
            await self._event(
                run_id,
                "security.active.failed",
                {"target": target.name, "session_id": None, "error_type": type(exc).__name__},
            )
            return "failed", None, self._active_metrics("failed", None), audit_error

        # The tool owns create/wait/result; emit the started/completed pair once
        # the session id is known so both events carry the real id.
        await self._event(
            run_id,
            "security.active.started",
            {"target": target.name, "session_id": output.session_id},
        )
        audit_error = audit_error or output.audit_error
        summary = summarize_active(
            output.reports,
            session_id=output.session_id,
            raw_result_path=output.raw_result_path,
        )
        metrics = self._active_metrics("completed", summary.score)
        metrics.update(
            {
                **summary.metrics,
                "dataset": output.datasets,
                "num_prompts": output.num_prompts,
                "raw_result_path": output.raw_result_path,
            }
        )
        await self._event(
            run_id,
            "security.active.completed",
            {
                "session_id": output.session_id,
                "asr": summary.metrics.get("asr"),
                "findings": len(summary.findings),
            },
        )
        return "completed", summary, metrics, audit_error

    def _result(
        self,
        passive,
        active: tuple[ActiveStatus, ActiveSummary | None, dict[str, Any], bool],
        spec_count: int,
    ) -> DimensionResult:
        active_status, active_summary, active_metrics, audit_error = active
        passive_score = passive.score
        active_score = active_summary.score if active_summary is not None else None
        severity_counts = active_metrics.get("severity_counts") or {}
        cap = 0.2 if severity_counts.get("critical") else (0.5 if severity_counts.get("high") else None)
        if passive_score is not None:
            score = min(passive_score, cap) if cap is not None else passive_score
        else:
            score = active_score

        passive_complete = passive.metrics.get("expected_outcomes", 0) > 0 and passive.metrics.get(
            "scored_outcomes", 0
        ) == passive.metrics.get("expected_outcomes")
        if passive_score is None and active_score is None:
            status = "failed"
        elif passive_score is not None and passive_complete and active_status in {"completed", "not_applicable"}:
            status = "ok"
        else:
            status = "partial"

        if status == "failed":
            summary = "security 被动与主动检查都没有可用结果"
            error = "no usable security score"
        elif status == "ok":
            summary = f"security 完成：被动 {passive_score:.3f}，主动 {active_status}"
            error = None
        else:
            passive_label = passive_score if passive_score is not None else "n/a"
            summary = f"security 部分完成：被动 {passive_label}，主动 {active_status}"
            if active_metrics.get("refused_reason"):
                summary += f"（拒绝原因：{active_metrics['refused_reason']}）"
            error = None

        findings: list[Finding] = list(passive.findings)
        if active_summary is not None:
            findings.extend(active_summary.findings)
        confidence = passive.confidence if passive_score is not None else (1.0 if active_status == "completed" else 0.0)

        return DimensionResult(
            dimension=Dimension.SECURITY,
            status=status,
            score=score,
            confidence=confidence,
            summary=summary,
            metrics={
                "passive": passive.metrics,
                "active": active_metrics,
                "score_cap": cap,
                "audit_error": audit_error,
                "grader_count": spec_count,
            },
            findings=findings,
            evidence=passive.evidence,
            usage=passive.usage,
            error=error,
        )

    @staticmethod
    def _failed(message: str) -> DimensionResult:
        return DimensionResult(
            dimension=Dimension.SECURITY,
            status="failed",
            score=None,
            confidence=0.0,
            summary=message,
            usage=Usage(),
            error=message,
        )
