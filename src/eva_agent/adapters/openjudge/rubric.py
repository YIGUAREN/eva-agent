"""Generate L1 rubrics with OpenJudge before plan approval."""

from __future__ import annotations

from openjudge.generator.simple_rubric.rubric_generator import TaskBasedRubricGenerator
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.prompt_template import LanguageEnum


class RubricGenerationError(RuntimeError):
    """Raised when rubric generation returns nothing usable."""


class OpenJudgeRubricGenerator:
    def __init__(
        self,
        model: BaseChatModel,
        *,
        language: str = "zh",
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._language = LanguageEnum(language)
        self._max_retries = max_retries

    async def generate(
        self,
        *,
        task_description: str,
        sample_queries: list[str],
        scenario: str | None = None,
    ) -> list[str]:
        generator = TaskBasedRubricGenerator(
            model=self._model,
            task_description=task_description,
            scenario=scenario,
            language=self._language,
            # The upstream fallback treats [] as "use its defaults", so pass a
            # non-empty marker that is filtered out below and fails closed.
            default_rubrics=[""],
            max_retries=self._max_retries,
        )
        rubrics = await generator.generate(sample_queries=sample_queries or None)
        cleaned = [rubric.strip() for rubric in rubrics if rubric.strip()]
        if not cleaned:
            raise RubricGenerationError("rubric generation returned no rubrics")
        return cleaned
