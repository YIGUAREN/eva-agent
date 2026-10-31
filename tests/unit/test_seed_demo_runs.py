import importlib.util
from pathlib import Path

import pytest

from eva_agent.packages.run_bundle import ImportResult
from tests.support.runtime import fake_config


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "seed_demo_runs.py"
    spec = importlib.util.spec_from_file_location("seed_demo_runs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_seed_imports_every_bundle_in_name_order(monkeypatch, tmp_path) -> None:
    module = _module()
    (tmp_path / "b.zip").write_bytes(b"PK")
    (tmp_path / "a.zip").write_bytes(b"PK")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    seen: list[str] = []

    async def fake_import(*, config, archive, public):
        seen.append(f"{archive.name}:{public}")
        return ImportResult(run_id=archive.stem, status="imported", artifacts=1)

    monkeypatch.setattr(module, "import_run_bundle", fake_import)
    results = await module.seed(fake_config(tmp_path), tmp_path, public=True)

    assert seen == ["a.zip:True", "b.zip:True"]
    assert [item.run_id for item in results] == ["a", "b"]


async def test_seed_reports_a_missing_directory(tmp_path) -> None:
    module = _module()
    with pytest.raises(FileNotFoundError):
        await module.seed(fake_config(tmp_path), tmp_path / "missing", public=True)
