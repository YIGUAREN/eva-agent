import argparse
import json
from contextlib import asynccontextmanager

import pytest

from eva_agent import cli
from eva_agent.packages.manifest import PackageError, PackageManifest
from tests.support.runtime import fake_config


class _FakeManager:
    @classmethod
    @asynccontextmanager
    async def open(cls, config, **kwargs):
        yield cls()

    async def run_row(self, run_id):
        return "row"


@pytest.fixture
def fake_manifest() -> PackageManifest:
    return PackageManifest(
        kind="grader_package",
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        report_version=4,
        files={},
    )


def test_export_grader_package_parses_the_run_and_out(tmp_path):
    args = cli.build_parser().parse_args(
        ["export-grader-package", "run-1", "--out", str(tmp_path)]
    )
    assert args.command == "export-grader-package"
    assert args.run_id == "run-1"
    assert args.out == str(tmp_path)
    assert args.config == "config/config.yaml"


async def test_export_grader_package_prints_the_package_summary(
    monkeypatch, capsys, tmp_path, fake_manifest
):
    calls = {}

    async def fake_build(**kwargs):
        calls.update(kwargs)
        archive = tmp_path / "run-1.grader-package.zip"
        archive.write_bytes(b"zip")
        return archive, fake_manifest

    monkeypatch.setattr(cli, "build_grader_package", fake_build)
    monkeypatch.setattr(cli, "RunManager", _FakeManager)
    config = fake_config(tmp_path)
    code = await cli._export_grader_package(
        argparse.Namespace(run_id="run-1", out=str(tmp_path)), config
    )
    assert code == 0
    assert calls["run_row"] == "row"
    assert calls["out_dir"] == tmp_path
    assert calls["config"] is config
    payload = json.loads(capsys.readouterr().out)
    assert payload["replayable"] is True
    assert payload["package"].endswith("run-1.grader-package.zip")


def test_export_grader_package_reports_an_unwritable_out_without_a_traceback(
    monkeypatch, capsys, tmp_path
):
    out_file = tmp_path / "out-is-a-file"
    out_file.write_text("not a directory", encoding="utf-8")

    async def fake_build(**kwargs):
        # The real builder's first filesystem operation; raises FileExistsError
        # whose ``filename`` is the absolute output path.
        kwargs["out_dir"].mkdir(parents=True, exist_ok=True)
        raise AssertionError("unreachable")

    monkeypatch.setattr(cli, "build_grader_package", fake_build)
    monkeypatch.setattr(cli, "RunManager", _FakeManager)
    monkeypatch.setattr(cli, "load_config", lambda path: fake_config(tmp_path))

    code = cli.main(["export-grader-package", "run-1", "--out", str(out_file)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: cannot write grader package:")
    assert "Traceback" not in captured.err
    assert str(out_file) not in captured.err


def test_main_reports_a_package_error_without_a_traceback(monkeypatch, capsys):
    async def failing_dispatch(args):
        raise PackageError("unsafe package")

    monkeypatch.setattr(cli, "_dispatch", failing_dispatch)
    code = cli.main(["export-grader-package", "run-1", "--out", "out"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == "error: unsafe package"
    assert "Traceback" not in captured.err
