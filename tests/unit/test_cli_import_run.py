import json
from types import SimpleNamespace

from eva_agent import cli


async def test_import_run_prints_the_result(monkeypatch, tmp_path, capsys) -> None:
    class FakeManager:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def import_run_bundle(self, archive, *, public):
            assert public is True
            return SimpleNamespace(run_id="run-1", status="imported", artifacts=2)

    monkeypatch.setattr(cli, "_open_manager", lambda config: FakeManager())
    code = await cli._import_run(
        SimpleNamespace(bundle=str(tmp_path / "b.zip"), public=True),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "run-1",
        "status": "imported",
        "artifacts": 2,
    }
