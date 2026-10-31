import pytest
import yaml

from eva_agent.adapters.aig.body import build_agent_scan_body, build_skill_scan_body, response_transform
from eva_agent.config.settings import AigAgentScanSettings, AigSkillScanSettings, ModelConfig
from eva_agent.domain.target import TargetKind, TargetSpec

EVAL = ModelConfig(
    provider="openai-compatible",
    model="judge",
    base_url="http://judge.local/v1",
    api_key_env="JUDGE_KEY",
)


def agent_target() -> TargetSpec:
    return TargetSpec(
        name="support-agent",
        kind=TargetKind.AGENT_HTTP,
        provider="dify",
        model="agent",
        url="https://agent.example.com/chat?mode=full",
        method="POST",
        headers={"X-Trace": "1"},
        request_template={"message": "{{prompt}}"},
        response_path="$.choices[0].message.content",
    )


def test_agent_scan_body_matches_the_contract() -> None:
    body = build_agent_scan_body(
        target=agent_target(),
        headers={"X-Trace": "1", "Authorization": "Bearer secret"},
        eval_model=EVAL,
        eval_token="judge-token",
        scan=AigAgentScanSettings(),
    )

    assert body["type"] == "agent_scan"
    content = body["content"]
    assert content["eval_model"] == {
        "model": "judge",
        "token": "judge-token",
        "base_url": "http://judge.local/v1",
    }
    assert content["language"] == "en"
    assert content["skills"] == ["data-leakage-detection", "tool-abuse-detection"]
    assert content["prompt"] == "Assess this agent for security weaknesses."
    assert "attachments" not in content

    config = yaml.safe_load(content["agent_config"])
    assert config == {
        "targets": [
            {
                "id": "http",
                "config": {
                    "url": "https://agent.example.com",
                    "endpoint": "/chat?mode=full",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        "X-Trace": "1",
                        "Authorization": "Bearer secret",
                    },
                    "body": {"message": "{{prompt}}"},
                    "transform_response": "choices[0].message.content",
                },
            }
        ]
    }


def test_agent_config_omits_an_empty_endpoint() -> None:
    target = agent_target().model_copy(update={"url": "https://agent.example.com"})
    body = build_agent_scan_body(
        target=target,
        headers={},
        eval_model=EVAL,
        eval_token="judge-token",
        scan=AigAgentScanSettings(),
    )
    config = yaml.safe_load(body["content"]["agent_config"])["targets"][0]["config"]
    assert config["url"] == "https://agent.example.com"
    assert "endpoint" not in config
    assert config["headers"] == {"Content-Type": "application/json"}


def test_skill_scan_body_carries_the_upload_reference_as_a_string() -> None:
    body = build_skill_scan_body(
        eval_model=EVAL,
        eval_token="judge-token",
        file_url="demo_1793422813849140825_1793422813.zip",
        scan=AigSkillScanSettings(),
    )

    assert body == {
        "type": "skill_scan",
        "content": {
            "model": {"model": "judge", "token": "judge-token", "base_url": "http://judge.local/v1"},
            "language": "zh",
            "prompt": "Analyze this skill package for security risks.",
            "attachments": "demo_1793422813849140825_1793422813.zip",
        },
    }


def test_response_transform_strips_the_jsonpath_prefix() -> None:
    assert response_transform("$.answer") == "answer"
    assert response_transform("$.choices[0].message.content") == "choices[0].message.content"


def test_response_transform_rejects_expressions() -> None:
    with pytest.raises(ValueError, match="response_path is not a simple dotted path"):
        response_transform("$..reply")
