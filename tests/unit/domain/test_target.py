import pytest
from pydantic import ValidationError

from eva_agent.domain.target import TargetKind, TargetSpec, UsagePaths


def openai_target(**overrides):
    data = {
        "name": "demo",
        "kind": TargetKind.OPENAI_COMPATIBLE,
        "provider": "demo",
        "model": "demo-echo",
        "base_url": "http://127.0.0.1:8100/v1",
    }
    data.update(overrides)
    return TargetSpec(**data)


def test_openai_target_accepts_minimal_fields():
    target = openai_target()
    assert target.kind is TargetKind.OPENAI_COMPATIBLE
    assert target.method is None
    assert target.usage_paths is None


def test_openai_target_rejects_http_json_fields():
    with pytest.raises(ValidationError, match="must not set http_json fields"):
        openai_target(url="http://example.com")


def test_http_json_target_requires_template_and_path():
    with pytest.raises(ValidationError, match="requires url, request_template, response_path"):
        TargetSpec(
            name="x",
            kind=TargetKind.HTTP_JSON,
            provider="custom",
            model="m",
            url="https://example.com/chat",
            method="POST",
        )


def test_http_json_target_rejects_openai_fields():
    with pytest.raises(ValidationError, match="must not set openai_compatible fields"):
        TargetSpec(
            name="x",
            kind=TargetKind.HTTP_JSON,
            provider="custom",
            model="m",
            url="https://example.com/chat",
            method="POST",
            request_template={"q": "{{prompt}}"},
            response_path="$.answer",
            base_url="https://example.com/v1",
        )


def test_headers_and_header_env_must_not_overlap():
    with pytest.raises(ValidationError, match="must not share a key"):
        TargetSpec(
            name="x",
            kind=TargetKind.HTTP_JSON,
            provider="custom",
            model="m",
            url="https://example.com/chat",
            method="POST",
            request_template={"q": "{{prompt}}"},
            response_path="$.answer",
            headers={"Authorization": "literal"},
            header_env={"Authorization": "AUTH_TOKEN"},
        )


def test_usage_paths_requires_at_least_one_path():
    with pytest.raises(ValidationError, match="at least one token path"):
        UsagePaths()


def test_http_json_requires_prompt_placeholder():
    with pytest.raises(ValidationError, match="\\{\\{prompt\\}\\} placeholder"):
        TargetSpec(
            name="x",
            kind=TargetKind.HTTP_JSON,
            provider="custom",
            model="m",
            url="https://example.com/chat",
            method="POST",
            request_template={"q": "static"},
            response_path="$.answer",
        )


def test_get_template_rejects_nested_values():
    with pytest.raises(ValidationError, match="GET request_template values"):
        TargetSpec(
            name="x",
            kind=TargetKind.HTTP_JSON,
            provider="custom",
            model="m",
            url="https://example.com/chat",
            method="GET",
            request_template={"q": "{{prompt}}", "nested": {"a": 1}},
            response_path="$.answer",
        )
