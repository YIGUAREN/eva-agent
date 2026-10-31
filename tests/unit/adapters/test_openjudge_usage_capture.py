"""The judge-model wrapper must report provider token usage to the tracker."""

from types import SimpleNamespace
from typing import Any, cast

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openjudge.generator.simple_rubric.rubric_generator import RubricGenerationOutput

from eva_agent.adapters.openjudge.models import JsonModeOpenAIChatModel
from eva_agent.budget.usage import track_usage


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> ChatCompletion:
        self.calls.append(kwargs)
        return ChatCompletion(
            id="chatcmpl-usage",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content='{"rubrics": ["a"], "reason": "ok"}'),
                )
            ],
            created=0,
            model="judge-model",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )

    async def parse(self, **kwargs: Any) -> ChatCompletion:
        raise AssertionError("judge endpoints reject json_schema response_format")


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


async def test_judge_calls_record_the_provider_token_totals() -> None:
    completions = _FakeCompletions()
    model = JsonModeOpenAIChatModel(
        model="judge-model",
        api_key="sk-test",
        base_url="https://example.test/v1",
        client=cast(Any, _FakeClient(completions)),
    )

    with track_usage() as totals:
        await model.achat(
            messages=[{"role": "user", "content": "Return JSON"}],
            structured_model=RubricGenerationOutput,
        )

    assert (totals.input_tokens, totals.output_tokens) == (11, 7)
