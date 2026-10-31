"""Pure authorization decision for an AIG active scan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from eva_agent.config.settings import AigSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import ScanKind, TargetKind, TargetSpec
from eva_agent.security.target_policy import SecurityPolicy

ScanReason = Literal[
    "allowed",
    "not_target",
    "not_upload",
    "unsupported_target_kind",
    "unsupported_scan_kind",
    "disabled",
    "not_admin",
    "missing_authorization",
    "authorization_mismatch",
    "target_not_allowlisted",
    "missing_eval_model_key",
    "budget_exceeded",
    "timeout_too_short",
]


@dataclass(frozen=True)
class ScanDecision:
    allowed: bool
    reason: ScanReason
    scan_kind: ScanKind | None = None
    scan_base_url: str | None = None
    target_url: str | None = None
    authorization_ref: str | None = None
    max_wait_s: float | None = None
    scan_requests: int = 0


def _deny(reason: ScanReason, *, ref: str | None = None, scan_kind: ScanKind | None = None) -> ScanDecision:
    return ScanDecision(allowed=False, reason=reason, scan_kind=scan_kind, authorization_ref=ref)


def _origin(url: str) -> str:
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


class ScanPolicy:
    def __init__(
        self,
        policy: SecurityPolicy,
        *,
        aig: AigSettings,
        eval_model_available: bool,
    ) -> None:
        self._policy = policy
        self._aig = aig
        self._eval_model_available = eval_model_available

    def decide(
        self,
        *,
        target: TargetSpec | None,
        input_kind: InputKind,
        probe_count: int,
        budget: Budget,
        scan_kind: ScanKind = "model_redteam_report",
        requester_is_admin: bool | None = None,
        authorization_ref: str | None = None,
    ) -> ScanDecision:
        """Decide one scan. Per-run values override the static policy when given."""
        if scan_kind == "model_redteam_report":
            return self._decide_model(
                target=target,
                input_kind=input_kind,
                probe_count=probe_count,
                budget=budget,
                requester_is_admin=requester_is_admin,
                authorization_ref=authorization_ref,
            )
        if scan_kind == "agent_scan":
            return self._decide_agent(
                target=target,
                input_kind=input_kind,
                budget=budget,
                requester_is_admin=requester_is_admin,
                authorization_ref=authorization_ref,
            )
        if scan_kind == "skill_scan":
            return self._decide_skill(
                input_kind=input_kind,
                budget=budget,
                requester_is_admin=requester_is_admin,
                authorization_ref=authorization_ref,
            )
        return _deny("unsupported_scan_kind", scan_kind=scan_kind)

    def _ref(self, authorization_ref: str | None) -> str | None:
        return self._policy.authorization_ref if authorization_ref is None else authorization_ref

    def _admin(self, requester_is_admin: bool | None) -> bool:
        return self._policy.requester_is_admin if requester_is_admin is None else requester_is_admin

    def _gate(self, *, scan_kind: ScanKind, requester_is_admin: bool | None, ref: str | None) -> ScanDecision | None:
        if not self._policy.active_scan_enabled:
            return _deny("disabled", ref=ref, scan_kind=scan_kind)
        if not self._admin(requester_is_admin):
            return _deny("not_admin", ref=ref, scan_kind=scan_kind)
        if not ref:
            return _deny("missing_authorization", scan_kind=scan_kind)
        if not self._eval_model_available:
            return _deny("missing_eval_model_key", ref=ref, scan_kind=scan_kind)
        return None

    def _limits(
        self,
        *,
        scan_kind: ScanKind,
        scan_requests: int,
        budget: Budget,
        ref: str | None,
        scan_base_url: str | None = None,
        target_url: str | None = None,
    ) -> ScanDecision:
        if scan_requests > budget.max_target_requests:
            return ScanDecision(
                allowed=False,
                reason="budget_exceeded",
                scan_kind=scan_kind,
                scan_base_url=scan_base_url,
                target_url=target_url,
                authorization_ref=ref,
                scan_requests=scan_requests,
            )
        max_wait_s = min(self._aig.max_wait_s, budget.max_wall_clock_s)
        if max_wait_s < self._aig.min_wait_s:
            return ScanDecision(
                allowed=False,
                reason="timeout_too_short",
                scan_kind=scan_kind,
                scan_base_url=scan_base_url,
                target_url=target_url,
                authorization_ref=ref,
                scan_requests=scan_requests,
            )
        return ScanDecision(
            allowed=True,
            reason="allowed",
            scan_kind=scan_kind,
            scan_base_url=scan_base_url,
            target_url=target_url,
            authorization_ref=ref,
            max_wait_s=max_wait_s,
            scan_requests=scan_requests,
        )

    def _demo_url(self, target: TargetSpec, *, url: str, ref: str) -> ScanDecision | str:
        demo = next(
            (
                item
                for item in self._policy.demo_targets
                if item.name == target.name and _origin(url) == item.origin
            ),
            None,
        )
        if demo is not None:
            if ref != demo.name:
                return _deny("authorization_mismatch", ref=ref, scan_kind="agent_scan")
            return demo.aig_base_url or url
        if _origin(url) not in self._policy.allowlist:
            return _deny("target_not_allowlisted", ref=ref, scan_kind="agent_scan")
        return url

    def _decide_model(
        self,
        *,
        target: TargetSpec | None,
        input_kind: InputKind,
        probe_count: int,
        budget: Budget,
        requester_is_admin: bool | None,
        authorization_ref: str | None,
    ) -> ScanDecision:
        ref = self._ref(authorization_ref)
        if input_kind is not InputKind.TARGET or target is None:
            return _deny("not_target", ref=ref, scan_kind="model_redteam_report")
        if target.kind is not TargetKind.OPENAI_COMPATIBLE:
            return _deny("unsupported_target_kind", ref=ref, scan_kind="model_redteam_report")
        denied = self._gate(
            scan_kind="model_redteam_report",
            requester_is_admin=requester_is_admin,
            ref=ref,
        )
        if denied is not None:
            return denied
        assert target.base_url is not None
        demo = next(
            (
                item
                for item in self._policy.demo_targets
                if item.name == target.name and _origin(target.base_url) == item.origin
            ),
            None,
        )
        if demo is not None:
            if ref != demo.name:
                return _deny("authorization_mismatch", ref=ref, scan_kind="model_redteam_report")
            scan_base_url = demo.aig_base_url or target.base_url
        else:
            if _origin(target.base_url) not in self._policy.allowlist:
                return _deny("target_not_allowlisted", ref=ref, scan_kind="model_redteam_report")
            scan_base_url = target.base_url
        if not scan_base_url:
            return _deny("target_not_allowlisted", ref=ref, scan_kind="model_redteam_report")
        scan_requests = probe_count + self._aig.scan.num_prompts * len(self._aig.scan.datasets)
        return self._limits(
            scan_kind="model_redteam_report",
            scan_requests=scan_requests,
            budget=budget,
            ref=ref,
            scan_base_url=scan_base_url,
        )

    def _decide_agent(
        self,
        *,
        target: TargetSpec | None,
        input_kind: InputKind,
        budget: Budget,
        requester_is_admin: bool | None,
        authorization_ref: str | None,
    ) -> ScanDecision:
        ref = self._ref(authorization_ref)
        if input_kind is not InputKind.TARGET or target is None:
            return _deny("not_target", ref=ref, scan_kind="agent_scan")
        if target.kind is not TargetKind.AGENT_HTTP:
            return _deny("unsupported_target_kind", ref=ref, scan_kind="agent_scan")
        denied = self._gate(scan_kind="agent_scan", requester_is_admin=requester_is_admin, ref=ref)
        if denied is not None:
            return denied
        assert target.url is not None
        resolved = self._demo_url(target, url=target.url, ref=ref or "")
        if isinstance(resolved, ScanDecision):
            return resolved
        return self._limits(
            scan_kind="agent_scan",
            scan_requests=self._aig.agent_scan.num_prompts,
            budget=budget,
            ref=ref,
            target_url=resolved,
        )

    def _decide_skill(
        self,
        *,
        input_kind: InputKind,
        budget: Budget,
        requester_is_admin: bool | None,
        authorization_ref: str | None,
    ) -> ScanDecision:
        ref = self._ref(authorization_ref)
        if input_kind is not InputKind.UPLOAD:
            return _deny("not_upload", ref=ref, scan_kind="skill_scan")
        denied = self._gate(scan_kind="skill_scan", requester_is_admin=requester_is_admin, ref=ref)
        if denied is not None:
            return denied
        return self._limits(
            scan_kind="skill_scan",
            scan_requests=self._aig.skill_scan.num_prompts,
            budget=budget,
            ref=ref,
        )
