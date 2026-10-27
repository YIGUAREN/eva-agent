import pytest
from pydantic import ValidationError

from eva_agent.domain.grader import GraderSpec, RubricSpec


def test_builtin_grader_spec() -> None:
    spec = GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")

    assert spec.level == "builtin"
    assert spec.model_role == "judge"
    assert spec.input_map == {}


def test_llm_rubric_grader_spec_keeps_rubric() -> None:
    spec = GraderSpec(level="llm_rubric", name="helpfulness", mode="pointwise", rubric="回答必须直接解决问题")

    assert spec.rubric == "回答必须直接解决问题"


def test_grader_level_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        GraderSpec(level="python", name="x", mode="pointwise")  # type: ignore[arg-type]


def test_grader_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        GraderSpec(level="builtin", name="x", mode="batch", builtin_id="correctness")  # type: ignore[arg-type]


def test_rubric_weight_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RubricSpec(name="准确性", weight=0)
