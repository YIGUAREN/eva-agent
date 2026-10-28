"""Offline tests for the OpenJudge judge model factory."""

from typing import Literal

import pytest
from openjudge.models.openai_chat_model import OpenAIChatModel

from eva_agent.adapters.openjudge.models import build_openjudge_model
from eva_agent.config.settings import MissingSecretError, ModelConfig


def _model(provider: Literal["openai-compatible", "anthropic", "litellm"] = "openai-compatible") -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model="judge-model",
        base_url="https://example.test/v1",
        api_key_env="JUDGE_KEY",
    )


def test_builds_openai_compatible_model() -> None:
    built = build_openjudge_model(_model("openai-compatible"), env={"JUDGE_KEY": "sk-test"})

    assert isinstance(built, OpenAIChatModel)
    assert built.model == "judge-model"


def test_builds_litellm_proxy_model() -> None:
    built = build_openjudge_model(_model("litellm"), env={"JUDGE_KEY": "sk-test"})

    assert isinstance(built, OpenAIChatModel)


def test_rejects_anthropic_provider() -> None:
    anthropic = ModelConfig(
        provider="anthropic",
        model="judge-model",
        api_key_env="JUDGE_KEY",
    )

    with pytest.raises(ValueError, match="OpenAI-compatible"):
        build_openjudge_model(anthropic, env={"JUDGE_KEY": "sk-test"})


def test_missing_api_key_raises() -> None:
    with pytest.raises(MissingSecretError):
        build_openjudge_model(_model("openai-compatible"), env={})
