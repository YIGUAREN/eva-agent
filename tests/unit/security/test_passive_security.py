from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.domain.run import DatasetRef, InputKind
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.security.passive import has_code, passive_specs, summarize_passive


def evidence() -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.PASTE,
        pasted={"query": "q", "response": "a"},
        manifest_sha256="a" * 64,
    )


def target_evidence() -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.TARGET,
        target=TargetSpec(
            name="demo",
            kind=TargetKind.OPENAI_COMPATIBLE,
            provider="demo",
            model="demo-echo",
            base_url="http://127.0.0.1:8100/v1",
        ),
        target_prompts=["q"],
        manifest_sha256="a" * 64,
    )


def dataset_evidence() -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.DATASET,
        dataset=DatasetRef(path="data/qa.csv", format="csv"),
        manifest_sha256="a" * 64,
    )


def rows() -> list[dict[str, str]]:
    return [{"query": "q", "response": "a", "context": "", "reference": ""}]


def code_rows() -> list[dict[str, str]]:
    return [{"query": "q", "response": "def add(a, b):\n    return a + b", "context": "", "reference": ""}]


def test_harmfulness_always_runs_and_code_security_is_conditional() -> None:
    assert [(spec.name, spec.builtin_id) for spec in passive_specs(rows())] == [("harmfulness", "harmfulness")]
    assert [(spec.name, spec.builtin_id) for spec in passive_specs(code_rows())] == [
        ("harmfulness", "harmfulness"),
        ("code_security", "code_security"),
    ]
    assert has_code(code_rows()) is True
    assert has_code([{"query": "q", "response": "", "context": "#include <stdio.h>", "reference": ""}]) is True


def test_summarize_passive_scores_and_findings() -> None:
    output = GraderRunOutput(
        outcomes=[
            GraderSampleOutcome(grader="harmfulness", sample_index=0, score=0.2),
            GraderSampleOutcome(grader="code_security", sample_index=0, score=0.8),
            GraderSampleOutcome(grader="harmfulness", sample_index=1, score=1.0),
            GraderSampleOutcome(grader="code_security", sample_index=1, score=0.8),
        ],
        duration_ms=7,
    )
    rows_value = [*code_rows(), {**code_rows()[0], "response": "class A: pass"}]
    result = summarize_passive(output, specs=passive_specs(rows_value), rows=rows_value, evidence=evidence())
    assert result.score == 0.7
    assert result.confidence == 1.0
    assert result.metrics["by_grader"] == {"code_security": 0.8, "harmfulness": 0.6}
    assert result.metrics["by_sample"] == {"0": 0.5, "1": 0.9}
    assert result.metrics["scored_rows"] == 2
    assert result.metrics["total_rows"] == 2
    assert len(result.findings) == 1
    assert result.findings[0].severity == "high"
    assert result.findings[0].id == "security-passive-1-harmfulness"
    assert result.usage.grader_calls == 4


def test_summarize_passive_uses_target_probe_source_indexes() -> None:
    output = GraderRunOutput(outcomes=[GraderSampleOutcome(grader="harmfulness", sample_index=0, score=0.2)])
    result = summarize_passive(
        output,
        specs=passive_specs(rows()),
        rows=rows(),
        evidence=target_evidence(),
        source_indexes=[7],
    )
    assert len(result.findings) == 1
    assert result.findings[0].evidence[0].locator == "target_probe.jsonl#7"


def test_summarize_passive_uses_dataset_path_as_workspace_evidence() -> None:
    output = GraderRunOutput(outcomes=[GraderSampleOutcome(grader="harmfulness", sample_index=0, score=0.2)])
    result = summarize_passive(
        output,
        specs=passive_specs(rows()),
        rows=rows(),
        evidence=dataset_evidence(),
    )
    assert result.evidence[0].locator == "data/qa.csv"
    assert len(result.findings) == 1
    assert result.findings[0].evidence[0].locator == "data/qa.csv#0"


def test_summarize_passive_reports_all_failed() -> None:
    output = GraderRunOutput(
        outcomes=[
            GraderSampleOutcome(grader=spec.name, sample_index=0, score=None, error="grader unavailable")
            for spec in passive_specs(rows())
        ]
    )
    result = summarize_passive(output, specs=passive_specs(rows()), rows=rows(), evidence=evidence())
    assert result.score is None
    assert result.confidence == 0.0
    assert result.metrics["errors"] == ["grader unavailable"]
