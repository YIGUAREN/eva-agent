from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome, NormalizedQuality, Usage
from eva_agent.domain.run import InputKind
from eva_agent.security.scans import empty_passive, merge_passive, summarize_threat

_DIGEST = "0" * 64


def evidence() -> EvidenceBundle:
    return EvidenceBundle(kind=InputKind.UPLOAD, root="/tmp/run/evidence", manifest_sha256=_DIGEST)


def threat_output(scores: list[float | None]) -> GraderRunOutput:
    return GraderRunOutput(
        outcomes=[
            GraderSampleOutcome(
                grader="threat_analysis",
                sample_index=index,
                score=score,
                raw_score=None if score is None else 1 + score * 3,
                error=None if score is not None else "boom",
            )
            for index, score in enumerate(scores)
        ],
        duration_ms=10,
    )


def test_threat_summary_marks_low_scoring_packages() -> None:
    rows = [{"skill_name": "greeter"}, {"skill_name": "ssh-collector"}]
    summary = summarize_threat(threat_output([1.0, 0.0]), rows=rows, evidence=evidence())

    assert summary.score == 0.5
    assert summary.metrics["by_grader"] == {"threat_analysis": 0.5}
    assert summary.metrics["skill_packages"] == 2
    assert [finding.title for finding in summary.findings] == ["skill 包 ssh-collector 的威胁分析未通过"]


def test_threat_summary_without_scores_is_empty() -> None:
    summary = summarize_threat(threat_output([None]), rows=[{"skill_name": "greeter"}], evidence=evidence())
    assert summary.score is None
    assert summary.metrics["errors"] == ["boom"]


def test_merge_keeps_the_lower_score_and_adds_graders() -> None:
    passive = NormalizedQuality(
        score=0.9,
        confidence=1.0,
        metrics={"by_grader": {"harmfulness": 0.9}, "scored_outcomes": 1, "expected_outcomes": 1, "errors": []},
        usage=Usage(grader_calls=1, samples=1),
    )
    threat = summarize_threat(
        threat_output([1.0, 0.0]),
        rows=[{"skill_name": "a"}, {"skill_name": "b"}],
        evidence=evidence(),
    )

    merged = merge_passive(passive, threat)

    assert merged.score == 0.5
    assert set(merged.metrics["by_grader"]) == {"harmfulness", "threat_analysis"}
    assert merged.metrics["skill_packages"] == 2
    assert merged.metrics["expected_outcomes"] == 3
    assert merged.usage.grader_calls == 3


def test_threat_summary_keeps_the_cache_counts() -> None:
    output = GraderRunOutput(
        outcomes=[GraderSampleOutcome(grader="threat_analysis", sample_index=0, score=1.0)],
        cached=2,
        fresh=1,
    )

    summary = summarize_threat(output, rows=[{"skill_name": "greeter"}], evidence=evidence())

    assert (summary.usage.cached, summary.usage.fresh) == (2, 1)


def test_merge_sums_the_cache_counts() -> None:
    passive = NormalizedQuality(
        score=0.9,
        confidence=1.0,
        metrics={"by_grader": {}, "scored_outcomes": 0, "expected_outcomes": 0, "errors": []},
        usage=Usage(cached=2, fresh=1),
    )
    threat = NormalizedQuality(
        score=0.8,
        confidence=1.0,
        metrics={"by_grader": {}, "scored_outcomes": 0, "expected_outcomes": 0, "errors": []},
        usage=Usage(cached=3, fresh=4),
    )

    merged = merge_passive(passive, threat)

    assert (merged.usage.cached, merged.usage.fresh) == (5, 5)


def test_merge_without_threat_returns_the_passive_summary() -> None:
    passive = empty_passive(evidence())
    assert merge_passive(passive, None) is passive
