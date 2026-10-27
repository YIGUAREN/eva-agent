"""Build LangChain chat models and AIG eval-model payloads from one config shape."""

from __future__ import annotations

from collections.abc import Mapping

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from eva_agent.config.settings import ModelConfig, resolve_api_key


def build_agent_chat_model(model: ModelConfig, env: Mapping[str, str] | None = None) -> BaseChatModel:
    """Build the chat model used by the agent graph."""
    api_key = SecretStr(resolve_api_key(model, env))

    if model.provider == "anthropic":
        # Pyright synthesizes the alias signature and treats aliased optional
        # fields as required, hence the explicit None timeout/stop.
        return ChatAnthropic(
            model_name=model.model,
            api_key=api_key,
            base_url=model.base_url,
            temperature=0,
            timeout=None,
            stop=None,
        )

    # Both "openai-compatible" and "litellm" reach an OpenAI-shaped endpoint.
    return ChatOpenAI(model=model.model, api_key=api_key, base_url=model.base_url, temperature=0)


def build_aig_eval_model(model: ModelConfig, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the eval_model payload expected by AI-Infra-Guard."""
    return {
        "model": model.model,
        "token": resolve_api_key(model, env),
        "base_url": model.base_url or "",
    }
