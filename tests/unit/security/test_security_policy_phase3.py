import pytest
from pydantic import ValidationError

from eva_agent.security.target_policy import DemoTarget, SecurityPolicy


def test_policy_is_closed_by_default() -> None:
    policy = SecurityPolicy()
    assert policy.active_scan_enabled is False
    assert policy.requester_is_admin is False
    assert policy.authorization_ref is None
    assert policy.allowlist == []


def test_demo_target_accepts_scanner_base_url() -> None:
    target = DemoTarget(
        name="demo",
        origin="http://127.0.0.1:8100",
        aig_base_url="http://demo-target:8100/v1",
    )
    assert target.aig_base_url == "http://demo-target:8100/v1"


def test_demo_target_rejects_credentials_in_scanner_url() -> None:
    with pytest.raises(ValidationError):
        DemoTarget(name="demo", origin="http://127.0.0.1:8100", aig_base_url="http://u:p@h:1/v1")


def test_allowlist_is_validated_and_deduplicated() -> None:
    policy = SecurityPolicy(allowlist=["https://api.example.com:443", "https://api.example.com:443"])
    assert policy.allowlist == ["https://api.example.com:443"]
    with pytest.raises(ValidationError):
        SecurityPolicy(allowlist=["https://api.example.com/v1"])


def test_origin_errors_do_not_echo_the_url() -> None:
    with pytest.raises(ValueError) as exc:
        SecurityPolicy(allowlist=["http://user:secret@example.com:80"])
    assert "secret" not in str(exc.value)
