from typing import cast

from eva_agent.config.settings import AigAgentScanSettings, AigScanSettings, AigSettings, AigSkillScanSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import ScanKind, TargetKind, TargetSpec
from eva_agent.security.scan_policy import ScanPolicy
from eva_agent.security.target_policy import DemoTarget, SecurityPolicy


def aig() -> AigSettings:
    return AigSettings(
        scan=AigScanSettings(datasets=["d"], num_prompts=2, random_seed=1),
        agent_scan=AigAgentScanSettings(num_prompts=3),
        skill_scan=AigSkillScanSettings(num_prompts=1),
    )


def policy(**overrides) -> SecurityPolicy:
    payload = {
        "active_scan_enabled": True,
        "requester_is_admin": True,
        "authorization_ref": "demo-agent",
        "demo_targets": [DemoTarget(name="demo-agent", origin="http://127.0.0.1:8200")],
        "allowlist": [],
    }
    payload.update(overrides)
    return SecurityPolicy(**payload)


def agent_target() -> TargetSpec:
    return TargetSpec(
        name="demo-agent",
        kind=TargetKind.AGENT_HTTP,
        provider="dify",
        model="agent",
        url="http://127.0.0.1:8200/chat",
        method="POST",
        request_template={"message": "{{prompt}}"},
        response_path="$.answer",
    )


def budget(max_requests: int = 10, wall: float = 100) -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=wall, max_target_requests=max_requests)


def decide(scan_policy: ScanPolicy, *, scan_kind: str, target: TargetSpec | None = None, **kwargs):
    return scan_policy.decide(
        target=target,
        input_kind=InputKind.TARGET if target is not None else InputKind.UPLOAD,
        scan_kind=cast(ScanKind, scan_kind),
        probe_count=0,
        budget=kwargs.pop("budget", budget()),
        **kwargs,
    )


def test_agent_scan_allowed_for_demo_agent() -> None:
    decision = decide(
        ScanPolicy(policy(), aig=aig(), eval_model_available=True),
        scan_kind="agent_scan",
        target=agent_target(),
    )
    assert decision.allowed is True
    assert decision.scan_kind == "agent_scan"
    assert decision.target_url == "http://127.0.0.1:8200/chat"
    assert decision.scan_requests == 3


def test_agent_scan_requires_agent_target() -> None:
    openai = TargetSpec(
        name="demo-agent",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="m",
        base_url="http://127.0.0.1:8200/v1",
    )
    decision = decide(
        ScanPolicy(policy(), aig=aig(), eval_model_available=True),
        scan_kind="agent_scan",
        target=openai,
    )
    assert decision.allowed is False
    assert decision.reason == "unsupported_target_kind"


def test_agent_scan_denied_for_non_allowlisted_origin() -> None:
    target = agent_target().model_copy(update={"url": "https://agent.example.com/chat"})
    decision = decide(ScanPolicy(policy(), aig=aig(), eval_model_available=True), scan_kind="agent_scan", target=target)
    assert decision.reason == "target_not_allowlisted"


def test_agent_scan_authorization_mismatch() -> None:
    decision = decide(
        ScanPolicy(policy(), aig=aig(), eval_model_available=True),
        scan_kind="agent_scan",
        target=agent_target(),
        authorization_ref="other",
    )
    assert decision.reason == "authorization_mismatch"


def test_skill_scan_allowed_for_upload() -> None:
    decision = decide(
        ScanPolicy(policy(), aig=aig(), eval_model_available=True),
        scan_kind="skill_scan",
        requester_is_admin=True,
        authorization_ref="skill-approval-1",
    )
    assert decision.allowed is True
    assert decision.scan_kind == "skill_scan"
    assert decision.scan_requests == 1


def test_skill_scan_requires_upload() -> None:
    decision = ScanPolicy(policy(), aig=aig(), eval_model_available=True).decide(
        target=None,
        input_kind=InputKind.PASTE,
        scan_kind="skill_scan",
        probe_count=0,
        budget=budget(),
    )
    assert decision.reason == "not_upload"


def test_skill_scan_rejects_target_backed_evidence() -> None:
    decision = ScanPolicy(policy(), aig=aig(), eval_model_available=True).decide(
        target=agent_target(),
        input_kind=InputKind.TARGET,
        scan_kind="skill_scan",
        probe_count=0,
        budget=budget(),
    )
    assert decision.reason == "not_upload"


def test_per_run_admin_override_denies_when_run_is_not_admin() -> None:
    decision = decide(
        ScanPolicy(policy(), aig=aig(), eval_model_available=True),
        scan_kind="agent_scan",
        target=agent_target(),
        requester_is_admin=False,
    )
    assert decision.reason == "not_admin"


def test_run_authorization_ref_overrides_config() -> None:
    decision = decide(
        ScanPolicy(policy(), aig=aig(), eval_model_available=True),
        scan_kind="agent_scan",
        target=agent_target(),
        authorization_ref="run-approval-9",
    )
    assert decision.reason == "authorization_mismatch"
    assert decision.authorization_ref == "run-approval-9"


def test_agent_scan_budget_exceeded_before_calling_aig() -> None:
    decision = decide(
        ScanPolicy(policy(), aig=aig(), eval_model_available=True),
        scan_kind="agent_scan",
        target=agent_target(),
        budget=budget(max_requests=2),
    )
    assert decision.allowed is False
    assert decision.reason == "budget_exceeded"
    assert decision.scan_requests == 3


def test_unknown_scan_kind_is_refused() -> None:
    decision = decide(ScanPolicy(policy(), aig=aig(), eval_model_available=True), scan_kind="mcp_scan")
    assert decision.reason == "unsupported_scan_kind"
