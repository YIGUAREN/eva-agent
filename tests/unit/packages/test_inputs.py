import json
from pathlib import Path

from eva_agent.adapters.inputs.dataset import write_dataset_workspace
from eva_agent.adapters.inputs.paste import write_paste_workspace
from eva_agent.config.settings import AppConfig, PricingSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind, RunInput
from eva_agent.packages.inputs import collect_package_input
from eva_agent.packages.manifest import sha256_bytes
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import RunRow
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import dataset_input, fake_config, paste_input


def _budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1)


def _priced_config(tmp_path: Path) -> AppConfig:
    """A fake config whose pricing table matches the fake model."""
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "models:\n"
        "  - provider: openai-compatible\n"
        "    model: fake\n"
        "    input_per_million: 1.0\n"
        "    output_per_million: 2.0\n",
        encoding="utf-8",
    )
    return fake_config(tmp_path).model_copy(update={"pricing": PricingSettings(path=str(pricing))})


async def _run_row(tmp_path: Path, run_input: RunInput) -> RunRow:
    config = fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    repo = RunRepository(create_session_factory(engine))
    await repo.create("run-1", run_input, budget=_budget())
    row = await repo.get("run-1")
    assert row is not None
    await engine.dispose()
    return row


async def test_paste_input_collects_the_single_row(tmp_path) -> None:
    config = _priced_config(tmp_path)
    workspace = Path(config.storage.data_dir) / "runs" / "run-1" / "evidence"
    workspace.mkdir(parents=True)
    (workspace / "paste.json").write_text(
        json.dumps({"query": "q", "response": "a", "context": ""}), encoding="utf-8"
    )
    run_input = paste_input("q", "a")
    assert run_input.paste is not None
    write_paste_workspace("run-1", run_input.paste, config.storage.data_dir)
    row = await _run_row(tmp_path, run_input)
    payload = await collect_package_input(config=config, run_row=row)
    assert payload.run_input.kind is InputKind.PASTE
    assert payload.rows == [{"query": "q", "response": "a", "context": "", "reference": ""}]
    assert payload.source_indexes == [None]
    assert payload.evidence.kind is InputKind.PASTE
    assert payload.evidence.manifest_sha256 == sha256_bytes((workspace / "manifest.json").read_bytes())
    # A paste run has no target cost to price, so no price rows are packaged.
    assert payload.prices == []


async def test_dataset_input_samples_the_declared_limit(tmp_path) -> None:
    config = fake_config(tmp_path)
    dataset = tmp_path / "rows.csv"
    dataset.write_text("query,response\nq1,a1\nq2,a2\n", encoding="utf-8")
    run_input = dataset_input(dataset)
    assert run_input.dataset is not None
    write_dataset_workspace(
        "run-1",
        run_input.dataset,
        config.storage.data_dir,
        max_bytes=config.pipeline.dataset_max_bytes,
        max_rows=config.pipeline.dataset_max_rows,
    )
    row = await _run_row(tmp_path, run_input)
    payload = await collect_package_input(config=config, run_row=row)
    assert [item["query"] for item in payload.rows] == ["q1", "q2"]
    assert payload.source_indexes == [None, None]
