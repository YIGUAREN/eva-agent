from pathlib import Path

from eva_agent.runtime.artifacts import run_paths
from eva_agent.worker.runner import _artifact_specs


def specs(tmp_path: Path) -> dict[str, tuple[str, str]]:
    return {
        name: (visibility, content_type)
        for name, _path, visibility, content_type in _artifact_specs(run_paths(tmp_path, "run-1"))
    }


def test_aig_results_are_registered_as_admin_artifacts(tmp_path: Path) -> None:
    paths = run_paths(tmp_path, "run-1")
    paths.aig_dir.mkdir(parents=True)
    (paths.aig_dir / "sess-abc.json").write_text("{}", encoding="utf-8")
    (paths.aig_dir / "sess-abc.csv").write_text("id,score\n1,0\n", encoding="utf-8")

    registered = specs(tmp_path)

    assert registered["aig/sess-abc.json"] == ("admin", "application/json")
    assert registered["aig/sess-abc.csv"] == ("admin", "text/csv; charset=utf-8")


def test_report_artifacts_are_still_registered(tmp_path: Path) -> None:
    registered = specs(tmp_path)
    assert registered["report.json"] == ("admin", "application/json")
    assert registered["report.md"] == ("admin", "text/markdown; charset=utf-8")
    assert not [name for name in registered if name.startswith("aig/")]


def test_unexpected_aig_files_are_ignored(tmp_path: Path) -> None:
    paths = run_paths(tmp_path, "run-1")
    paths.aig_dir.mkdir(parents=True)
    (paths.aig_dir / "notes.txt").write_text("x", encoding="utf-8")
    (paths.aig_dir / "nested").mkdir()
    (paths.aig_dir / "nested" / "sess-abc.json").write_text("{}", encoding="utf-8")
    (paths.aig_dir / "bad name.json").write_text("{}", encoding="utf-8")

    assert not [name for name in specs(tmp_path) if name.startswith("aig/")]
