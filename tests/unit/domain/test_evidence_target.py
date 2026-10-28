import pytest
from pydantic import ValidationError

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.result import EvidenceRef
from eva_agent.domain.run import InputKind
from eva_agent.domain.target import TargetKind, TargetSpec


def demo_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )


def test_target_evidence_requires_target():
    with pytest.raises(ValidationError, match="target evidence requires a target"):
        EvidenceBundle(kind=InputKind.TARGET, manifest_sha256="a" * 64)


def test_non_target_evidence_rejects_target():
    with pytest.raises(ValidationError, match="target is only valid for target evidence"):
        EvidenceBundle(kind=InputKind.PASTE, target=demo_target(), manifest_sha256="a" * 64)


def test_target_evidence_round_trip():
    bundle = EvidenceBundle(
        kind=InputKind.TARGET,
        target=demo_target(),
        target_prompts=["q1"],
        manifest_sha256="a" * 64,
    )

    assert EvidenceBundle.model_validate(bundle.model_dump(mode="json")) == bundle
    assert EvidenceRef(kind="target_request", locator="target_probe.jsonl#0").kind == "target_request"
