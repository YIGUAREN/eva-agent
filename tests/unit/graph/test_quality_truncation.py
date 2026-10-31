from eva_agent.domain.result import NormalizedQuality, Usage
from eva_agent.graph.experts.quality import summarize_quality


def test_truncated_quality_without_a_score_is_partial_not_failed() -> None:
    normalized = NormalizedQuality(
        score=None,
        confidence=0.0,
        metrics={},
        findings=[],
        evidence=[],
        usage=Usage(),
        truncated_reason="budget",
    )
    result = summarize_quality(normalized, sample_count=5, spec_count=2, truncated_reason="budget")
    assert result.status == "partial"
    assert result.truncated_reason == "budget"
    assert result.score is None
