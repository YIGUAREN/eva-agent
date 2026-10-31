import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

from eva_agent import cli
from eva_agent.domain.run import RunStatus
from eva_agent.packages.manifest import FileEntry, PackageError, PackageManifest
from eva_agent.packages.run_bundle import build_run_bundle
from eva_agent.storage.models import RunRow
from eva_agent.storage.reports import ReportSnapshot
from tests.support.runtime import fake_config


async def test_export_run_prints_the_archive_path(monkeypatch, tmp_path, capsys) -> None:
    archive = tmp_path / "run-1.run-bundle.zip"
    archive.write_bytes(b"PK")
    manifest = PackageManifest(
        kind="run_bundle",
        run_id="run-1",
        created_at="2026-01-01T00:00:00+00:00",
        replayable=False,
        files={"run.json": FileEntry(sha256="a" * 64, size=2)},
    )

    class FakeManager:
        def __init__(self) -> None:
            self.ran: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def export_run_bundle(self, run_id, out_dir):
            self.ran.append(run_id)
            return archive, manifest

    manager = FakeManager()
    monkeypatch.setattr(cli, "_open_manager", lambda config: manager)
    code = await cli._export_run(
        SimpleNamespace(run_id="run-1", out=str(tmp_path / "out")),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
    )
    assert code == 0
    assert manager.ran == ["run-1"]
    assert json.loads(capsys.readouterr().out)["package"].endswith("run-1.run-bundle.zip")


class _FailingManager:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def export_run_bundle(self, run_id, out_dir):
        raise self._error


class _RealBundleManager:
    """Run the real bundle builder so the CLI sees the genuine filesystem error."""

    def __init__(self, config) -> None:
        self._config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def export_run_bundle(self, run_id, out_dir):
        run_row = cast(
            RunRow,
            SimpleNamespace(
                id=run_id,
                status=RunStatus.COMPLETED.value,
                input_json={},
                budget_json={},
                plan_json={},
                is_public=True,
                created_at=datetime.now(UTC),
                published_at=None,
            ),
        )
        report = ReportSnapshot(
            run_id=run_id, version=1, payload={}, markdown="", sha256="0" * 64
        )
        return await build_run_bundle(
            config=self._config,
            run_row=run_row,
            report=report,
            artifacts=[],
            out_dir=out_dir,
        )


def _run_export_main(monkeypatch, tmp_path, manager) -> int:
    monkeypatch.setattr(cli, "load_config", lambda path: fake_config(tmp_path))
    monkeypatch.setattr(cli, "_open_manager", lambda config: manager)
    return cli.main(["export-run", "run-1", "--out", str(tmp_path / "out")])


def test_main_reports_an_unknown_run_without_a_traceback(monkeypatch, tmp_path, capsys) -> None:
    code = _run_export_main(monkeypatch, tmp_path, _FailingManager(KeyError("run-404")))
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == "error: unknown run 'run-404'"
    assert "Traceback" not in captured.err


def test_main_reports_an_export_package_error_without_a_traceback(
    monkeypatch, tmp_path, capsys
) -> None:
    code = _run_export_main(
        monkeypatch, tmp_path, _FailingManager(PackageError("only public runs can be exported"))
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == "error: only public runs can be exported"
    assert "Traceback" not in captured.err


def test_main_reports_an_unwritable_out_without_a_traceback(
    monkeypatch, tmp_path, capsys
) -> None:
    config = fake_config(tmp_path)
    out_file = tmp_path / "out"
    out_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "_open_manager", lambda ignored: _RealBundleManager(config))

    code = cli.main(["export-run", "run-1", "--out", str(out_file)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: cannot write run bundle:")
    assert "Traceback" not in captured.err
    assert str(out_file) not in captured.err
