import pytest
from pydantic import ValidationError

from eva_agent.domain.target import TargetKind, TargetSpec, scan_kind_for


def agent(**overrides) -> TargetSpec:
    payload = {
        "name": "support-agent",
        "kind": TargetKind.AGENT_HTTP,
        "provider": "dify",
        "model": "agent",
        "url": "https://agent.example.com/chat",
        "method": "POST",
        "request_template": {"message": "{{prompt}}"},
        "response_path": "$.answer",
    }
    payload.update(overrides)
    return TargetSpec(**payload)


def test_agent_http_defaults_to_agent_scan() -> None:
    target = agent()
    assert scan_kind_for(target) == "agent_scan"
    assert scan_kind_for(agent(scan_kind="agent_scan")) == "agent_scan"


def test_agent_http_rejects_model_redteam_scan_kind() -> None:
    with pytest.raises(ValidationError, match="agent_http target only supports agent_scan"):
        agent(scan_kind="model_redteam_report")


def test_agent_http_rejects_openai_fields() -> None:
    with pytest.raises(ValidationError, match="agent_http target must not set openai_compatible fields"):
        agent(base_url="https://api.example.com/v1")


def test_openai_compatible_defaults_to_model_redteam() -> None:
    target = TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="m",
        base_url="https://api.example.com/v1",
    )
    assert scan_kind_for(target) == "model_redteam_report"


def test_openai_compatible_rejects_agent_scan() -> None:
    with pytest.raises(ValidationError, match="openai_compatible target only supports model_redteam_report"):
        TargetSpec(
            name="demo",
            kind=TargetKind.OPENAI_COMPATIBLE,
            provider="demo",
            model="m",
            base_url="https://api.example.com/v1",
            scan_kind="agent_scan",
        )


def test_http_json_has_no_scan_kind() -> None:
    target = TargetSpec(
        name="http",
        kind=TargetKind.HTTP_JSON,
        provider="p",
        model="m",
        url="https://api.example.com/run",
        method="POST",
        request_template={"q": "{{prompt}}"},
        response_path="$.out",
    )
    assert scan_kind_for(target) is None
    with pytest.raises(ValidationError, match="http_json target must not set scan_kind"):
        TargetSpec(
            name="http",
            kind=TargetKind.HTTP_JSON,
            provider="p",
            model="m",
            url="https://api.example.com/run",
            method="POST",
            request_template={"q": "{{prompt}}"},
            response_path="$.out",
            scan_kind="agent_scan",
        )


def test_agent_http_requires_http_json_fields() -> None:
    with pytest.raises(ValidationError, match="agent_http target requires url, request_template, response_path"):
        TargetSpec(
            name="bad",
            kind=TargetKind.AGENT_HTTP,
            provider="dify",
            model="agent",
            url="https://agent.example.com/chat",
        )
