import pytest

from eva_agent.domain.grader import CheckSpec, GraderSpec, ScoreSpec
from eva_agent.graders_dsl.schema import DeclarativeSpecError, validate_declarative_spec


def spec(**overrides) -> GraderSpec:
    data = {
        "level": "declarative",
        "name": "latency",
        "input_map": {"latency_ms": "$.latency_ms", "response": "$.response"},
        "checks": [CheckSpec(type="numeric_threshold", field="latency_ms", op="<=", value=3000)],
    }
    data.update(overrides)
    return GraderSpec(**data)


def test_valid_spec_passes():
    validate_declarative_spec(spec())


def test_field_must_be_declared():
    with pytest.raises(DeclarativeSpecError, match="field 'missing' is not declared"):
        validate_declarative_spec(spec(checks=[CheckSpec(type="json_parse", field="missing")]))


def test_duplicate_check_names_rejected():
    checks = [
        CheckSpec(name="same", type="json_parse", field="response"),
        CheckSpec(name="same", type="regex", field="response", pattern="x"),
    ]
    with pytest.raises(DeclarativeSpecError, match="duplicate check name"):
        validate_declarative_spec(spec(checks=checks))


def test_unknown_weight_key_rejected():
    with pytest.raises(DeclarativeSpecError, match="unknown weight key"):
        validate_declarative_spec(spec(score=ScoreSpec(weights={"missing": 1.0})))


def test_unknown_schema_keyword_rejected():
    checks = [CheckSpec(type="json_schema", field="response", schema={"type": "object", "$ref": "x"})]
    with pytest.raises(DeclarativeSpecError, match="unknown schema key"):
        validate_declarative_spec(spec(checks=checks))


def test_invalid_regex_rejected():
    checks = [CheckSpec(type="regex", field="response", pattern="[")]
    with pytest.raises(DeclarativeSpecError, match="invalid regex"):
        validate_declarative_spec(spec(checks=checks))


def test_invalid_input_map_path_rejected():
    with pytest.raises(DeclarativeSpecError, match="invalid input_map path"):
        validate_declarative_spec(spec(input_map={"latency_ms": "latency_ms"}))


def test_non_declarative_spec_rejected():
    with pytest.raises(DeclarativeSpecError, match="declarative grader"):
        validate_declarative_spec(GraderSpec(level="builtin", name="x", builtin_id="relevance"))
