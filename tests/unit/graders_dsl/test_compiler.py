import pytest

from eva_agent.domain.grader import CheckSpec, GraderSpec, ScoreSpec
from eva_agent.graders_dsl.compiler import compile_grader


def grader(checks, *, weights=None, input_map=None) -> GraderSpec:
    return GraderSpec(
        level="declarative",
        name="checks",
        input_map=input_map or {"response": "$.response", "latency_ms": "$.latency_ms", "status_code": "$.status_code"},
        checks=checks,
        score=None if weights is None else ScoreSpec(weights=weights),
    )


@pytest.mark.parametrize(
    ("check", "record", "expected"),
    [
        (CheckSpec(type="json_parse", field="response"), {"response": '{"a": 1}'}, 1.0),
        (CheckSpec(type="json_parse", field="response"), {"response": "not json"}, 0.0),
        (
            CheckSpec(
                type="json_schema",
                field="response",
                schema={"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}},
            ),
            {"response": '{"a": 1}'},
            1.0,
        ),
        (
            CheckSpec(type="json_schema", field="response", schema={"type": "array", "items": {"type": "string"}}),
            {"response": '["a", 1]'},
            0.0,
        ),
        (CheckSpec(type="regex", field="response", pattern="demo answer: .*"), {"response": "demo answer: hi"}, 1.0),
        (CheckSpec(type="regex", field="response", pattern="demo answer: .*"), {"response": "x demo answer: hi"}, 0.0),
        (CheckSpec(type="numeric_threshold", field="latency_ms", op="<=", value=100), {"latency_ms": 100}, 1.0),
        (CheckSpec(type="numeric_threshold", field="latency_ms", op="<", value=100), {"latency_ms": True}, 0.0),
        (CheckSpec(type="numeric_threshold", field="status_code", op="==", value=200), {"status_code": 200}, 1.0),
        (CheckSpec(type="numeric_threshold", field="status_code", op=">=", value=200), {}, 0.0),
    ],
)
def test_check_semantics(check, record, expected):
    score, results = compile_grader(grader([check])).evaluate(record)

    assert score == expected
    assert results[0].passed is (expected == 1.0)


def test_weighted_mean_uses_weights():
    checks = [
        CheckSpec(name="a", type="json_parse", field="response"),
        CheckSpec(name="b", type="numeric_threshold", field="latency_ms", op="<=", value=100),
    ]
    spec = grader(checks, weights={"a": 3.0, "b": 1.0})

    score, _ = compile_grader(spec).evaluate({"response": "bad", "latency_ms": 10})

    assert score == 0.25


def test_missing_field_fails_without_raising():
    score, results = compile_grader(grader([CheckSpec(type="json_parse", field="response")])).evaluate({})

    assert score == 0.0
    assert results[0].passed is False
    assert "missing" in results[0].reason
