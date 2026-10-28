from openjudge.graders.schema import GraderError, GraderRank, GraderScore

from eva_agent.adapters.openjudge.catalog import GraderEntry
from eva_agent.adapters.openjudge.normalize import normalize_results
from eva_agent.domain.grader import GraderSpec


def correctness_entry() -> GraderEntry:
    return GraderEntry(
        id="correctness",
        klass="openjudge.graders.common.correctness.CorrectnessGrader",  # pyright: ignore[reportCallIssue]
        mode="pointwise",
        inputs=["query", "response", "reference_response"],
        score_min=1,
        score_max=5,
    )


def test_scores_are_normalized_and_clamped() -> None:
    specs = [GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")]
    results = {
        "correctness": [
            GraderScore(name="correctness", score=5, reason="great"),
            GraderScore(name="correctness", score=3, reason="ok"),
            GraderScore(name="correctness", score=0, reason="low"),
        ]
    }

    outcomes = normalize_results(results, specs, {"correctness": correctness_entry()}, 3)

    assert [item.score for item in outcomes] == [1.0, 0.5, 0.0]
    assert [item.raw_score for item in outcomes] == [5.0, 3.0, 0.0]


def test_infinite_score_becomes_error() -> None:
    specs = [GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")]
    results = {"correctness": [GraderScore(name="correctness", score=float("inf"), reason="overflow")]}

    outcomes = normalize_results(results, specs, {"correctness": correctness_entry()}, 1)

    assert len(outcomes) == 1
    assert outcomes[0].score is None
    assert outcomes[0].raw_score is None
    assert outcomes[0].error is not None
    assert "non-finite score" in outcomes[0].error


def test_nan_score_becomes_error() -> None:
    specs = [GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")]
    results = {"correctness": [GraderScore(name="correctness", score=float("nan"), reason="undefined")]}

    outcomes = normalize_results(results, specs, {"correctness": correctness_entry()}, 1)

    assert len(outcomes) == 1
    assert outcomes[0].score is None
    assert outcomes[0].raw_score is None
    assert outcomes[0].error is not None
    assert "non-finite score" in outcomes[0].error


def test_errors_are_kept_per_sample() -> None:
    specs = [GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")]
    results = {"correctness": [GraderError(name="correctness", error="boom", reason="failed")]}

    outcomes = normalize_results(results, specs, {"correctness": correctness_entry()}, 1)

    assert outcomes[0].score is None
    assert outcomes[0].error == "boom"


def test_listwise_results_are_rejected() -> None:
    specs = [GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")]
    results = {"correctness": [GraderRank(name="correctness", rank=[1], reason="ranked")]}

    outcomes = normalize_results(results, specs, {"correctness": correctness_entry()}, 1)

    assert outcomes[0].error is not None
    assert "listwise" in outcomes[0].error


def test_missing_results_become_errors() -> None:
    specs = [GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")]

    outcomes = normalize_results({}, specs, {"correctness": correctness_entry()}, 2)

    assert [item.error for item in outcomes] == ["missing result", "missing result"]


def test_unknown_builtin_is_fail_closed() -> None:
    specs = [GraderSpec(level="builtin", name="nope", mode="pointwise", builtin_id="nope")]

    outcomes = normalize_results({}, specs, {}, 1)

    assert outcomes[0].error is not None
    assert "unknown grader" in outcomes[0].error


def test_llm_rubric_uses_zero_to_one_range() -> None:
    specs = [GraderSpec(level="llm_rubric", name="helpfulness", mode="pointwise", rubric="be helpful")]
    results = {"helpfulness": [GraderScore(name="helpfulness", score=1, reason="good")]}

    outcomes = normalize_results(results, specs, {}, 1)

    assert outcomes[0].score == 1.0
