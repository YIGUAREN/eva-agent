import json
from types import SimpleNamespace

from eva_agent import cli
from eva_agent.packages.manifest import FileEntry, PackageManifest


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
