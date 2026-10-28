"""Offline tests for the OpenJudge judge model factory."""

from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openjudge.generator.simple_rubric.rubric_generator import RubricGenerationOutput
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.response import ChatResponse

from eva_agent.adapters.openjudge.models import JsonModeOpenAIChatModel, build_openjudge_model
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


def test_judge_model_is_built_in_json_mode() -> None:
    built = build_openjudge_model(_model("openai-compatible"), env={"JUDGE_KEY": "sk-test"})

    assert isinstance(built, JsonModeOpenAIChatModel)


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.calls: list[dict[str, Any]] = []
        self._content = content

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self.calls.append(kwargs)
        return ChatCompletion(
            id="chatcmpl-test",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content=self._content),
                )
            ],
            created=0,
            model="judge-model",
            object="chat.completion",
        )

    async def parse(self, **kwargs: Any) -> ChatCompletion:
        raise AssertionError("judge endpoints reject json_schema response_format")


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


async def test_structured_output_requests_json_object_instead_of_json_schema() -> None:
    completions = _FakeCompletions('{"rubrics": ["a", "b"], "reason": "ok"}')
    model = JsonModeOpenAIChatModel(
        model="judge-model",
        api_key="sk-test",
        base_url="https://example.test/v1",
        client=cast(Any, _FakeClient(completions)),
    )

    response = cast(
        ChatResponse,
        await model.achat(
            messages=[{"role": "user", "content": "Return JSON"}],
            structured_model=RubricGenerationOutput,
        ),
    )

    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert response.parsed == {"rubrics": ["a", "b"], "reason": "ok"}


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
