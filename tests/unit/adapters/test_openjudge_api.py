"""Pin the OpenJudge surface the adapter relies on."""

import openjudge
from openjudge.generator.iterative_rubric.query_rubric_generator import POINTWISE_EVALUATION_TEMPLATE
from openjudge.generator.simple_rubric.rubric_generator import TaskBasedRubricGenerator
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.hallucination import HallucinationGrader
from openjudge.graders.common.instruction_following import InstructionFollowingGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.text.similarity import SimilarityGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GradingRunner


def test_pinned_openjudge_version() -> None:
    assert openjudge.__version__ == "0.2.4"


def test_adapter_import_surface_exists() -> None:
    classes = [
        CorrectnessGrader,
        HallucinationGrader,
        InstructionFollowingGrader,
        RelevanceGrader,
        LLMGrader,
        SimilarityGrader,
        TaskBasedRubricGenerator,
        OpenAIChatModel,
        GradingRunner,
    ]
    assert all(cls is not None for cls in classes)
    assert POINTWISE_EVALUATION_TEMPLATE is not None
