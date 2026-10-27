from eva_agent.adapters.inputs.paste import build_paste_evidence
from eva_agent.domain.run import InputKind, PastedInput


def test_build_paste_evidence_hashes_content() -> None:
    bundle = build_paste_evidence(PastedInput(query="q", response="a"))

    assert bundle.kind is InputKind.PASTE
    assert bundle.pasted == {"query": "q", "response": "a"}
    assert len(bundle.manifest_sha256) == 64


def test_same_input_produces_same_hash() -> None:
    first = build_paste_evidence(PastedInput(query="q", response="a"))
    second = build_paste_evidence(PastedInput(query="q", response="a"))

    assert first.manifest_sha256 == second.manifest_sha256


def test_different_input_produces_different_hash() -> None:
    first = build_paste_evidence(PastedInput(query="q", response="a"))
    second = build_paste_evidence(PastedInput(query="q", response="b"))

    assert first.manifest_sha256 != second.manifest_sha256
