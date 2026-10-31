import argparse
import json

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
