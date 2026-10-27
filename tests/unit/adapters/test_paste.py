from pathlib import Path

from eva_agent.adapters.inputs.paste import write_paste_workspace
from eva_agent.domain.run import InputKind, PastedInput


def test_paste_workspace_hashes_content(tmp_path: Path) -> None:
    bundle = write_paste_workspace("run-1", PastedInput(query="q", response="a"), tmp_path)

    assert bundle.kind is InputKind.PASTE
    assert bundle.pasted == {"query": "q", "response": "a"}
    assert len(bundle.manifest_sha256) == 64


def test_same_input_produces_same_hash(tmp_path: Path) -> None:
    first = write_paste_workspace("run-1", PastedInput(query="q", response="a"), tmp_path)
    second = write_paste_workspace("run-1", PastedInput(query="q", response="a"), tmp_path)

    assert first.manifest_sha256 == second.manifest_sha256


def test_different_input_produces_different_hash(tmp_path: Path) -> None:
    first = write_paste_workspace("run-1", PastedInput(query="q", response="a"), tmp_path)
    second = write_paste_workspace("run-1", PastedInput(query="q", response="b"), tmp_path)

    assert first.manifest_sha256 != second.manifest_sha256
