import pytest
from pydantic import ValidationError

from eva_agent.domain.grader import CheckSpec, GraderSpec, ScoreSpec


def test_check_spec_defaults_name_to_type():
    check = CheckSpec(type="numeric_threshold", field="latency_ms", op="<=", value=3000)
    assert check.effective_name == "numeric_threshold"


def test_check_spec_rejects_inapplicable_fields():
    with pytest.raises(ValidationError, match="json_parse"):
        CheckSpec(type="json_parse", field="response", pattern="x")


def test_check_spec_requires_type_fields():
    with pytest.raises(ValidationError, match="numeric_threshold requires op and value"):
        CheckSpec(type="numeric_threshold", field="latency_ms")
    with pytest.raises(ValidationError, match="json_schema requires schema"):
        CheckSpec(type="json_schema", field="response")


def test_score_spec_rejects_non_positive_weight():
    with pytest.raises(ValidationError):
        ScoreSpec(weights={"a": 0})


def test_declarative_grader_requires_checks():
    with pytest.raises(ValidationError, match="requires checks"):
        GraderSpec(level="declarative", name="x")


def test_non_declarative_grader_rejects_checks():
    with pytest.raises(ValidationError, match="only valid for declarative"):
        GraderSpec(
            level="builtin",
            name="x",
            builtin_id="relevance",
            checks=[CheckSpec(type="json_parse", field="response")],
        )
