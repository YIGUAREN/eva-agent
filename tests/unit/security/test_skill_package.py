import asyncio
import json
from pathlib import Path

from eva_agent.domain.evidence import EvidenceBundle, EvidenceFile
from eva_agent.domain.run import InputKind
from eva_agent.security.skill_package import build_skill_rows, skill_spec
from eva_agent.tools.evidence import build_read_evidence_tool
from eva_agent.tools.registry import ToolRegistry

_DIGEST = "0" * 64


def registry() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(build_read_evidence_tool(1_000_000))
    return tools


def bundle(root: Path, *paths: str) -> EvidenceBundle:
    return EvidenceBundle(
        kind=InputKind.UPLOAD,
        root=str(root),
        files=[EvidenceFile(path=path, size=(root / path).stat().st_size, sha256=_DIGEST) for path in paths],
        manifest_sha256=_DIGEST,
    )


def write_package(root: Path) -> list[str]:
    (root / "SKILL.md").write_text("# Greeter\n\nSay hello.\n", encoding="utf-8")
    (root / "skill.yaml").write_text("name: greeter\nversion: 0.1.0\n", encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "references").mkdir()
    (root / "references" / "notes.md").write_text("benign\n", encoding="utf-8")
    return ["SKILL.md", "skill.yaml", "scripts/run.py", "references/notes.md"]


def rows(root: Path, paths: list[str]) -> list[dict]:
    return asyncio.run(build_skill_rows(registry(), bundle(root, *paths), run_id="run-1"))


def test_package_row_carries_the_five_grader_inputs(tmp_path: Path) -> None:
    paths = write_package(tmp_path)
    prepared = rows(tmp_path, paths)
    assert len(prepared) == 1
    row = prepared[0]
    assert set(row) == {
        "skill_name",
        "skill_manifest",
        "instruction_body",
        "script_contents",
        "reference_contents",
    }
    assert row["skill_name"] == tmp_path.name
    assert "name: greeter" in row["skill_manifest"]
    assert "Say hello." in row["instruction_body"]
    assert row["script_contents"] == ["print('hi')\n"]
    assert row["reference_contents"] == ["benign\n"]


def test_nested_example_package_is_a_second_row(tmp_path: Path) -> None:
    paths = write_package(tmp_path)
    (tmp_path / "examples" / "other").mkdir(parents=True)
    (tmp_path / "examples" / "other" / "SKILL.md").write_text("# Other\n", encoding="utf-8")
    prepared = rows(tmp_path, [*paths, "examples/other/SKILL.md"])
    assert len(prepared) == 2
    names = sorted(row["skill_name"] for row in prepared)
    assert names == sorted([tmp_path.name, "other"])
    other = next(row for row in prepared if row["skill_name"] == "other")
    assert other["instruction_body"] == "# Other\n"
    assert other["script_contents"] == []


def test_skill_manifest_without_instruction_body(tmp_path: Path) -> None:
    (tmp_path / "skill.yaml").write_text("name: bare\n", encoding="utf-8")
    prepared = rows(tmp_path, ["skill.yaml"])
    assert prepared[0]["instruction_body"] == ""
    assert prepared[0]["skill_manifest"] == "name: bare\n"


def test_large_files_are_truncated(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("a" * 400_000, encoding="utf-8")
    prepared = rows(tmp_path, ["SKILL.md"])
    assert len(prepared[0]["instruction_body"]) == 131_072


def test_skill_spec_is_a_builtin_threat_analysis_grader() -> None:
    spec = skill_spec()
    assert spec.level == "builtin"
    assert spec.builtin_id == "threat_analysis"
    assert spec.name == "threat_analysis"


def test_rows_are_json_serializable(tmp_path: Path) -> None:
    prepared = rows(tmp_path, write_package(tmp_path))
    assert json.loads(json.dumps(prepared)) == prepared
