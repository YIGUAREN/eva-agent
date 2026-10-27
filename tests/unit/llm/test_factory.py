import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from eva_agent.config.settings import MissingSecretError, ModelConfig
from eva_agent.llm.factory import build_agent_chat_model, build_aig_eval_model


def openai_config() -> ModelConfig:
    return ModelConfig(
        provider="openai-compatible",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
    )


def anthropic_config() -> ModelConfig:
    return ModelConfig(
        provider="anthropic",
        model="claude-sonnet-5-5",
        base_url=None,
        api_key_env="ANTHROPIC_API_KEY",
    )


def litellm_config() -> ModelConfig:
    return ModelConfig(
        provider="litellm",
        model="deepseek-chat",
        base_url="http://localhost:4000/v1",
        api_key_env="LITELLM_API_KEY",
    )


def test_build_litellm_chat_model_uses_base_url() -> None:
    model = build_agent_chat_model(litellm_config(), {"LITELLM_API_KEY": "sk-test"})

    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "http://localhost:4000/v1"


def test_build_openai_compatible_chat_model() -> None:
    model = build_agent_chat_model(openai_config(), {"DEEPSEEK_API_KEY": "sk-test"})

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "deepseek-chat"


def test_build_anthropic_chat_model() -> None:
    model = build_agent_chat_model(anthropic_config(), {"ANTHROPIC_API_KEY": "sk-ant-test"})

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-5-5"


def test_missing_secret_raises() -> None:
    with pytest.raises(MissingSecretError):
        build_agent_chat_model(openai_config(), {})


def test_build_aig_eval_model() -> None:
    payload = build_aig_eval_model(openai_config(), {"DEEPSEEK_API_KEY": "sk-test"})

    assert payload == {
        "model": "deepseek-chat",
        "token": "sk-test",
        "base_url": "https://api.deepseek.com/v1",
    }
