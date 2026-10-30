"""Pure authorization decision for an AIG active scan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from eva_agent.config.settings import AigSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.security.target_policy import SecurityPolicy

ScanReason = Literal[
    "allowed",
    "not_target",
    "unsupported_target_kind",
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
    scan_base_url: str | None = None
    authorization_ref: str | None = None
    max_wait_s: float | None = None
    scan_requests: int = 0


def _deny(reason: ScanReason, *, ref: str | None = None) -> ScanDecision:
    return ScanDecision(allowed=False, reason=reason, authorization_ref=ref)


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
    ) -> ScanDecision:
        ref = self._policy.authorization_ref
        if input_kind is not InputKind.TARGET or target is None:
            return _deny("not_target")
        if target.kind is not TargetKind.OPENAI_COMPATIBLE:
            return _deny("unsupported_target_kind", ref=ref)
        if not self._policy.active_scan_enabled:
            return _deny("disabled", ref=ref)
        if not self._policy.requester_is_admin:
            return _deny("not_admin", ref=ref)
        if not ref:
            return _deny("missing_authorization")

        demo = next(
            (
                item
                for item in self._policy.demo_targets
                if item.name == target.name and target.base_url is not None and _origin(target.base_url) == item.origin
            ),
            None,
        )
        if demo is not None:
            if ref != demo.name:
                return _deny("authorization_mismatch", ref=ref)
            scan_base_url = demo.aig_base_url or target.base_url
        else:
            if target.base_url is None or _origin(target.base_url) not in self._policy.allowlist:
                return _deny("target_not_allowlisted", ref=ref)
            scan_base_url = target.base_url

        if not scan_base_url:
            return _deny("target_not_allowlisted", ref=ref)
        if not self._eval_model_available:
            return _deny("missing_eval_model_key", ref=ref)

        scan_requests = probe_count + self._aig.scan.num_prompts * len(self._aig.scan.datasets)
        if scan_requests > budget.max_target_requests:
            return ScanDecision(
                allowed=False,
                reason="budget_exceeded",
                authorization_ref=ref,
                scan_requests=scan_requests,
            )
        max_wait_s = min(self._aig.max_wait_s, budget.max_wall_clock_s)
        if max_wait_s < self._aig.min_wait_s:
            return ScanDecision(
                allowed=False,
                reason="timeout_too_short",
                authorization_ref=ref,
                scan_requests=scan_requests,
            )
        return ScanDecision(
            allowed=True,
            reason="allowed",
            scan_base_url=scan_base_url,
            authorization_ref=ref,
            max_wait_s=max_wait_s,
            scan_requests=scan_requests,
        )
