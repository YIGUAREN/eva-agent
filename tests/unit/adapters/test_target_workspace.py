import json
from pathlib import Path

import pytest

from eva_agent.adapters.inputs.paste import write_paste_workspace
from eva_agent.adapters.inputs.target import write_target_workspace
from eva_agent.domain.run import InputKind, PastedInput
from eva_agent.domain.target import TargetKind, TargetSpec


def demo_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
    )


def test_target_workspace_writes_target_json_and_manifest(tmp_path: Path):
    bundle = write_target_workspace("run-1", demo_target(), ["q1", "q2"], tmp_path)

    evidence_dir = tmp_path / "runs" / "run-1" / "evidence"
    payload = json.loads((evidence_dir / "target.json").read_text(encoding="utf-8"))
    manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))

    assert payload == {"target": demo_target().model_dump(mode="json"), "prompts": ["q1", "q2"]}
    assert manifest["kind"] == "target"
    assert manifest["target"]["name"] == "demo"
    assert manifest["target_prompts"] == ["q1", "q2"]
    assert bundle.kind is InputKind.TARGET
    assert bundle.target == demo_target()
    assert bundle.target_prompts == ["q1", "q2"]
    assert bundle.root == str(evidence_dir)
    assert bundle.files[0].path == "target.json"


def test_target_workspace_hash_is_deterministic(tmp_path: Path):
    first = write_target_workspace("run-1", demo_target(), ["q1"], tmp_path)
    second = write_target_workspace("run-1", demo_target(), ["q1"], tmp_path)

    assert first.manifest_sha256 == second.manifest_sha256


def test_paste_manifest_has_null_target(tmp_path: Path):
    write_paste_workspace("run-2", PastedInput(query="q"), tmp_path)

    manifest = json.loads((tmp_path / "runs" / "run-2" / "evidence" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["target"] is None
    assert manifest["target_prompts"] == []


def test_target_workspace_rejects_path_like_run_id(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid run_id"):
        write_target_workspace("../../escape", demo_target(), ["q"], tmp_path / "data")
