"""Build the OpenJudge judge model; this is the only place that resolves its key."""

from __future__ import annotations

from collections.abc import Mapping

from openjudge.models.openai_chat_model import OpenAIChatModel

from eva_agent.config.settings import ModelConfig, resolve_api_key


def build_openjudge_model(model: ModelConfig, env: Mapping[str, str] | None = None) -> OpenAIChatModel:
    if model.provider == "anthropic":
        raise ValueError("OpenJudge requires an OpenAI-compatible judge endpoint")
    return OpenAIChatModel(
        model=model.model,
        api_key=resolve_api_key(model, env),
        base_url=model.base_url,
    )
