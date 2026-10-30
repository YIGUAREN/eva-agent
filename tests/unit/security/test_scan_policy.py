# tests/unit/security/test_scan_policy.py
import pytest

from eva_agent.config.settings import AigScanSettings, AigSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.security.scan_policy import ScanDecision, ScanPolicy
from eva_agent.security.target_policy import DemoTarget, SecurityPolicy


def aig() -> AigSettings:
    return AigSettings(scan=AigScanSettings(datasets=["d"], num_prompts=2, random_seed=1))


def policy(
    *,
    active_scan_enabled: bool = True,
    requester_is_admin: bool = True,
    authorization_ref: str | None = "demo",
    demo_targets: list[DemoTarget] | None = None,
    allowlist: list[str] | None = None,
) -> SecurityPolicy:
    return SecurityPolicy(
        active_scan_enabled=active_scan_enabled,
        requester_is_admin=requester_is_admin,
        authorization_ref=authorization_ref,
        demo_targets=(
            [DemoTarget(name="demo", origin="http://127.0.0.1:8100", aig_base_url="http://demo-target:8100/v1")]
            if demo_targets is None
            else demo_targets
        ),
        allowlist=[] if allowlist is None else allowlist,
    )


def demo() -> TargetSpec:
    return TargetSpec(
        name="demo", kind=TargetKind.OPENAI_COMPATIBLE, provider="demo", model="m",
        base_url="http://127.0.0.1:8100/v1",
    )


_DEFAULT_TARGET = demo()


def public() -> TargetSpec:
    return TargetSpec(
        name="public", kind=TargetKind.OPENAI_COMPATIBLE, provider="p", model="m",
        base_url="https://api.example.com/v1",
    )


def budget(max_requests: int = 10, wall: float = 100) -> Budget:
    return Budget(max_cost_usd=1, max_wall_clock_s=wall, max_target_requests=max_requests)


def decide(
    policy_value: SecurityPolicy,
    target: TargetSpec | None = _DEFAULT_TARGET,
    *,
    input_kind: InputKind = InputKind.TARGET,
    probe_count: int = 2,
    eval_available: bool = True,
    budget_value: Budget | None = None,
) -> ScanDecision:
    return ScanPolicy(policy_value, aig=aig(), eval_model_available=eval_available).decide(
        target=target, input_kind=input_kind, probe_count=probe_count, budget=budget_value or budget()
    )


def test_demo_scan_is_allowed_and_uses_container_url() -> None:
    decision = decide(policy())
    assert decision.allowed is True
    assert decision.reason == "allowed"
    assert decision.scan_base_url == "http://demo-target:8100/v1"
    assert decision.scan_requests == 4
    assert decision.max_wait_s == 100


@pytest.mark.parametrize(
    ("policy_value", "target", "kind", "probe", "eval_available", "budget_value", "reason"),
    [
        (policy(), demo(), InputKind.PASTE, 2, True, None, "not_target"),
        (policy(), None, InputKind.TARGET, 2, True, None, "not_target"),
        (policy(), public(), InputKind.TARGET, 2, True, None, "target_not_allowlisted"),
        (policy(active_scan_enabled=False), demo(), InputKind.TARGET, 2, True, None, "disabled"),
        (policy(requester_is_admin=False), demo(), InputKind.TARGET, 2, True, None, "not_admin"),
        (policy(authorization_ref=None), demo(), InputKind.TARGET, 2, True, None, "missing_authorization"),
        (policy(authorization_ref="wrong"), demo(), InputKind.TARGET, 2, True, None, "authorization_mismatch"),
        (policy(), demo(), InputKind.TARGET, 2, False, None, "missing_eval_model_key"),
        (policy(), demo(), InputKind.TARGET, 9, True, None, "budget_exceeded"),
        (policy(), demo(), InputKind.TARGET, 2, True, budget(wall=10), "timeout_too_short"),
    ],
)
def test_matrix(policy_value, target, kind, probe, eval_available, budget_value, reason) -> None:
    decision = decide(
        policy_value, target, input_kind=kind, probe_count=probe,
        eval_available=eval_available, budget_value=budget_value,
    )
    assert decision.allowed is False
    assert decision.reason == reason


def test_allowlisted_public_origin_is_allowed() -> None:
    decision = decide(
        policy(demo_targets=[], allowlist=["https://api.example.com:443"]),
        public(), probe_count=0,
    )
    assert decision.allowed is True
    assert decision.scan_base_url == "https://api.example.com/v1"


def test_demo_name_with_a_different_origin_is_not_the_demo_target() -> None:
    target = demo().model_copy(update={"base_url": "http://127.0.0.1:9999/v1"})
    decision = decide(policy(), target)
    assert decision.allowed is False
    assert decision.reason == "target_not_allowlisted"


def test_http_json_is_rejected() -> None:
    target = TargetSpec(
        name="json", kind=TargetKind.HTTP_JSON, provider="p", model="m",
        url="https://api.example.com/x", method="POST",
        request_template={"p": "{{prompt}}"}, response_path="$.text",
    )
    assert decide(policy(), target).reason == "unsupported_target_kind"
