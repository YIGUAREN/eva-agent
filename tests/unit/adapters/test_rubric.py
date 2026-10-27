import pytest

from eva_agent.adapters.openjudge.rubric import OpenJudgeRubricGenerator, RubricGenerationError
from tests.fakes.openjudge import FailingChatModel, FakeChatModel


async def test_generator_returns_rubrics() -> None:
    generator = OpenJudgeRubricGenerator(FakeChatModel({"rubrics": ["a", " b "], "reason": "ok"}))

    rubrics = await generator.generate(task_description="评估回答", sample_queries=["q"])

    assert rubrics == ["a", "b"]


async def test_empty_generation_is_fail_closed() -> None:
    generator = OpenJudgeRubricGenerator(FakeChatModel({"rubrics": [], "reason": "none"}))

    with pytest.raises(RubricGenerationError, match="no rubrics"):
        await generator.generate(task_description="评估回答", sample_queries=["q"])


async def test_model_failure_does_not_use_default_rubrics() -> None:
    generator = OpenJudgeRubricGenerator(FailingChatModel(), max_retries=1)

    with pytest.raises(RubricGenerationError, match="no rubrics"):
        await generator.generate(task_description="评估回答", sample_queries=["q"])
