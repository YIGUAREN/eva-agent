"""Build the OpenJudge judge model; this is the only place that resolves its key."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from openai import AsyncOpenAI
from openjudge.models.openai_chat_model import OpenAIChatModel

from eva_agent.budget.usage import record_usage
from eva_agent.config.settings import ModelConfig, resolve_api_key


class _JsonObjectCompletions:
    """Turn structured-output calls into `response_format=json_object` calls.

    OpenJudge asks for structured output with a json_schema response format.
    OpenAI-compatible endpoints such as DeepSeek reject that with HTTP 400 while
    accepting json_object; OpenJudge already parses the JSON body when the SDK
    returns no `parsed` field, so no schema validation happens here.
    """

    def __init__(self, completions: Any) -> None:
        self._completions = completions

    async def parse(self, **kwargs: Any) -> Any:
        kwargs["response_format"] = {"type": "json_object"}
        response = await self._completions.create(**kwargs)
        record_usage(response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _JsonObjectChat:
    def __init__(self, chat: Any) -> None:
        self._chat = chat
        self.completions = _JsonObjectCompletions(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _JsonObjectClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _JsonObjectChat(client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class JsonModeOpenAIChatModel(OpenAIChatModel):
    """OpenJudge judge model that requests JSON objects instead of JSON schemas."""

    def __init__(self, *, client: AsyncOpenAI | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if client is not None:
            self.client = client
        self.client = cast(Any, _JsonObjectClient(self.client))


def build_openjudge_model(model: ModelConfig, env: Mapping[str, str] | None = None) -> OpenAIChatModel:
    if model.provider == "anthropic":
        raise ValueError("OpenJudge requires an OpenAI-compatible judge endpoint")
    return JsonModeOpenAIChatModel(
        model=model.model,
        api_key=resolve_api_key(model, env),
        base_url=model.base_url,
    )
