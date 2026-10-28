import pytest

from eva_agent.adapters.target.secrets import (
    MissingSecretError,
    resolve_headers,
    validate_target_env,
)
from eva_agent.domain.target import TargetKind, TargetSpec


def openai_target(api_key_env: str | None = None) -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="m",
        base_url="https://api.example.com/v1",
        api_key_env=api_key_env,
    )


def json_target() -> TargetSpec:
    return TargetSpec(
        name="generic",
        kind=TargetKind.HTTP_JSON,
        provider="custom",
        model="m",
        url="https://api.example.com/chat",
        method="POST",
        headers={"X-Static": "yes"},
        header_env={"Authorization": "EXTERNAL_AUTH"},
        request_template={"q": "{{prompt}}"},
        response_path="$.answer",
    )


def test_openai_key_becomes_bearer_header():
    headers = resolve_headers(openai_target("OPENAI_KEY"), {"OPENAI_KEY": "sk-test"})

    assert headers == {"Authorization": "Bearer sk-test"}


def test_openai_without_key_env_has_no_auth():
    assert resolve_headers(openai_target(), {}) == {}


def test_json_target_merges_literal_and_env_headers():
    headers = resolve_headers(json_target(), {"EXTERNAL_AUTH": "token"})

    assert headers == {"X-Static": "yes", "Authorization": "token"}


def test_missing_env_value_rejected():
    with pytest.raises(MissingSecretError, match="OPENAI_KEY"):
        validate_target_env(openai_target("OPENAI_KEY"), {})
    with pytest.raises(MissingSecretError, match="EXTERNAL_AUTH"):
        resolve_headers(json_target(), {})
