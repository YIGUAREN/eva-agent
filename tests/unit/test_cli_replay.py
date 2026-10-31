import argparse
import json

import pytest

from eva_agent import cli
from eva_agent.packages.replay import Difference, ReplayResult


def _args(tmp_path):
    return argparse.Namespace(package=str(tmp_path / "p.zip"), out=str(tmp_path / "out"))


async def test_replay_writes_a_matching_result(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "read_grader_package", lambda path: "package")
    monkeypatch.setattr(
        cli,
        "replay_package",
        lambda package: ReplayResult(matched=True, compared=["quality"], skipped=[], differences=[]),
    )
    assert await cli._replay(_args(tmp_path), None) == 0
    payload = json.loads((tmp_path / "out" / "replay.json").read_text(encoding="utf-8"))
    assert payload["matched"] is True
    assert json.loads(capsys.readouterr().out)["compared"] == ["quality"]


async def test_replay_exits_non_zero_on_a_difference(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "read_grader_package", lambda path: "package")
    monkeypatch.setattr(
        cli,
        "replay_package",
        lambda package: ReplayResult(
            matched=False,
            compared=["quality"],
            skipped=[],
            differences=[Difference("dimensions.quality.score", 0.8, 0.1)],
        ),
    )
    assert await cli._replay(_args(tmp_path), None) == 1
    payload = json.loads((tmp_path / "out" / "replay.json").read_text(encoding="utf-8"))
    assert payload["matched"] is False
    assert payload["differences"] == [
        {"path": "dimensions.quality.score", "recorded": 0.8, "replayed": 0.1}
    ]


@pytest.mark.parametrize("kind", ["out-is-a-file", "replay-json-is-a-directory"])
def test_main_reports_an_unwritable_out_without_a_traceback(
    monkeypatch, tmp_path, capsys, kind
) -> None:
    monkeypatch.setattr(cli, "read_grader_package", lambda path: "package")
    monkeypatch.setattr(cli, "replay_package", lambda package: ReplayResult(matched=True))
    out_dir = tmp_path / "out"
    if kind == "out-is-a-file":
        out_dir.write_text("not a directory", encoding="utf-8")
    else:
        (out_dir / "replay.json").mkdir(parents=True)
    code = cli.main(["replay", "package.zip", "--out", str(out_dir)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: cannot write replay result:")
    assert "Traceback" not in captured.err
    assert str(out_dir) not in captured.err


def test_replay_never_loads_config(monkeypatch, tmp_path, capsys) -> None:
    def fail(path):
        raise AssertionError("replay must not load config")

    monkeypatch.setattr(cli, "load_config", fail)
    code = cli.main(["replay", str(tmp_path / "missing.zip"), "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err
