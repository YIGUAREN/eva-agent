import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eva_agent.adapters.inputs.dataset import write_dataset_workspace
from eva_agent.adapters.inputs.paste import write_paste_workspace
from eva_agent.adapters.inputs.target import write_target_workspace
from eva_agent.config.settings import AppConfig, PricingSettings
from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan, SkippedDimension
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import InputKind, RunInput
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.packages.inputs import (
    EvidenceSnapshot,
    _error_detail,
    collect_package_input,
    load_evidence_snapshot,
    load_probe_records,
)
from eva_agent.packages.manifest import PackageError, sha256_bytes
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import RunRow
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import budget, dataset_input, fake_config, paste_input

_DEFAULT_SAMPLE_SIZE = 50


def _priced_config(tmp_path: Path, body: str | None = None) -> AppConfig:
    """A fake config whose pricing table matches the fake model and one unused model."""
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        body
        or (
            "models:\n"
            "  - provider: openai-compatible\n"
            "    model: fake\n"
            "    input_per_million: 1.0\n"
            "    output_per_million: 2.0\n"
            "  - provider: other\n"
            "    model: unused\n"
            "    input_per_million: 9.0\n"
            "    output_per_million: 9.0\n"
        ),
        encoding="utf-8",
    )
    return fake_config(tmp_path).model_copy(update={"pricing": PricingSettings(path=str(pricing))})


def _quality_plan(*, sample_size: int | None) -> EvalPlan:
    return EvalPlan(
        objective="评估回答质量",
        evidence_summary="测试证据",
        dimensions=[
            DimensionPlan(
                dimension=Dimension.QUALITY,
                enabled=True,
                sample_size=sample_size,
                graders=[GraderSpec(level="builtin", name="correctness", builtin_id="correctness")],
            ),
            DimensionPlan(dimension=Dimension.SECURITY, enabled=True, sample_size=None),
        ],
        skipped=[SkippedDimension(dimension=Dimension.LATENCY, reason="未启用")],
        budget=budget(),
    )


async def _run_row(
    tmp_path: Path,
    run_input: RunInput,
    *,
    plan: EvalPlan | None = None,
    config: AppConfig | None = None,
) -> RunRow:
    config = config or fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    repo = RunRepository(create_session_factory(engine))
    await repo.create("run-1", run_input, budget=_budget())
    if plan is not None:
        await repo.set_plan("run-1", plan)
    row = await repo.get("run-1")
    assert row is not None
    await engine.dispose()
    return row


def _budget() -> Budget:
    return Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1)


def _write_dataset(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    dataset = tmp_path / "rows.csv"
    lines = ["query,response", *(f"{query},{response}" for query, response in rows)]
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset


def _write_target_run(config: AppConfig, target: TargetSpec, prompts: list[str]) -> RunInput:
    workspace = write_target_workspace("run-1", target, prompts, config.storage.data_dir)
    assert workspace.root is not None
    run_root = Path(workspace.root).parent
    record = ProbeRecord(
        index=0,
        prompt=prompts[0],
        started_at=datetime.now(UTC),
        status_code=200,
        latency_ms=5,
        response_text="hello",
        input_tokens=1,
        output_tokens=2,
        usage_source="target",
    )
    (run_root / "target_probe.jsonl").write_text(record.model_dump_json() + "\n", encoding="utf-8")
    return RunInput(kind=InputKind.TARGET, target_ref=target.name, target_prompts=prompts)


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
    assert payload.dimension_limits == {}


async def test_dataset_input_samples_the_declared_limit(tmp_path) -> None:
    config = fake_config(tmp_path)
    dataset = _write_dataset(tmp_path, [("q1", "a1"), ("q2", "a2")])
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


async def test_dataset_payload_keeps_only_the_evidence_relative_name(tmp_path) -> None:
    config = fake_config(tmp_path)
    dataset = _write_dataset(tmp_path, [("q1", "a1")])
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
    assert payload.run_input.dataset is not None
    assert payload.run_input.dataset.path == "dataset.csv"
    assert str(dataset) not in json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)


async def test_null_sample_sizes_use_the_expert_default(tmp_path) -> None:
    config = fake_config(tmp_path)
    dataset = _write_dataset(tmp_path, [(f"q{index}", f"a{index}") for index in range(60)])
    run_input = dataset_input(dataset)
    assert run_input.dataset is not None
    write_dataset_workspace(
        "run-1",
        run_input.dataset,
        config.storage.data_dir,
        max_bytes=config.pipeline.dataset_max_bytes,
        max_rows=config.pipeline.dataset_max_rows,
    )
    plan = _quality_plan(sample_size=None)
    row = await _run_row(tmp_path, run_input, plan=plan)
    payload = await collect_package_input(config=config, run_row=row)
    assert payload.dimension_limits == {
        Dimension.QUALITY.value: _DEFAULT_SAMPLE_SIZE,
        Dimension.SECURITY.value: _DEFAULT_SAMPLE_SIZE,
    }
    assert len(payload.rows) == _DEFAULT_SAMPLE_SIZE


async def test_mixed_sample_sizes_use_the_declared_limit_and_the_default(tmp_path) -> None:
    config = fake_config(tmp_path)
    dataset = _write_dataset(tmp_path, [(f"q{index}", f"a{index}") for index in range(60)])
    run_input = dataset_input(dataset)
    assert run_input.dataset is not None
    write_dataset_workspace(
        "run-1",
        run_input.dataset,
        config.storage.data_dir,
        max_bytes=config.pipeline.dataset_max_bytes,
        max_rows=config.pipeline.dataset_max_rows,
    )
    plan = _quality_plan(sample_size=5)
    row = await _run_row(tmp_path, run_input, plan=plan)
    payload = await collect_package_input(config=config, run_row=row)
    assert payload.dimension_limits == {
        Dimension.QUALITY.value: 5,
        Dimension.SECURITY.value: _DEFAULT_SAMPLE_SIZE,
    }
    assert len(payload.rows) == _DEFAULT_SAMPLE_SIZE


async def test_target_headers_never_enter_the_package_input(tmp_path) -> None:
    config = _priced_config(
        tmp_path,
        "models:\n"
        "  - provider: demo\n"
        "    model: demo-echo\n"
        "    input_per_million: 1.0\n"
        "    output_per_million: 2.0\n",
    )
    target = TargetSpec(
        name="demo",
        kind=TargetKind.HTTP_JSON,
        provider="demo",
        model="demo-echo",
        url="http://127.0.0.1:8100/chat",
        method="POST",
        headers={"x-tenant": "acme-prod"},
        header_env={"authorization": "DEMO_API_KEY"},
        request_template={"prompt": "{{prompt}}"},
        response_path="$.text",
    )
    run_input = _write_target_run(config, target, ["hi"])
    row = await _run_row(tmp_path, run_input)
    payload = await collect_package_input(config=config, run_row=row)
    assert payload.evidence.target is not None
    assert payload.evidence.target.headers == {}
    assert payload.evidence.target.header_env == {}
    serialized = json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)
    assert "acme-prod" not in serialized
    assert "x-tenant" not in serialized
    assert "DEMO_API_KEY" not in serialized
    assert "authorization" not in serialized


async def test_target_run_packages_only_the_price_row_it_used(tmp_path) -> None:
    config = _priced_config(tmp_path)
    target = TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="openai-compatible",
        model="fake",
        base_url="http://127.0.0.1:8100/v1",
    )
    run_input = _write_target_run(config, target, ["hi"])
    row = await _run_row(tmp_path, run_input)
    payload = await collect_package_input(config=config, run_row=row)
    assert [(item.provider, item.model) for item in payload.prices] == [("openai-compatible", "fake")]


async def test_target_run_fails_closed_when_its_price_row_is_missing(tmp_path) -> None:
    config = _priced_config(
        tmp_path,
        "models:\n"
        "  - provider: other\n"
        "    model: unused\n"
        "    input_per_million: 9.0\n"
        "    output_per_million: 9.0\n",
    )
    target = TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="openai-compatible",
        model="fake",
        base_url="http://127.0.0.1:8100/v1",
    )
    run_input = _write_target_run(config, target, ["hi"])
    row = await _run_row(tmp_path, run_input)
    with pytest.raises(PackageError, match="cannot load pricing"):
        await collect_package_input(config=config, run_row=row)


def test_load_evidence_snapshot_missing_manifest_omits_the_path(tmp_path) -> None:
    with pytest.raises(PackageError) as info:
        load_evidence_snapshot(tmp_path / "evidence")
    assert str(tmp_path) not in str(info.value)


def test_load_evidence_snapshot_does_not_echo_invalid_input(tmp_path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "paste",
                "files": [],
                "pasted": {"query": ["SECRET-VALUE"]},
                "dataset": None,
                "target": None,
                "target_prompts": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PackageError) as info:
        load_evidence_snapshot(evidence)
    assert "SECRET-VALUE" not in str(info.value)


def test_load_probe_records_rejects_malformed_json_without_a_path(tmp_path) -> None:
    run_root = tmp_path / "run-1"
    run_root.mkdir()
    (run_root / "target_probe.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(PackageError) as info:
        load_probe_records(run_root)
    assert str(tmp_path) not in str(info.value)


def test_error_detail_hides_validation_inputs_and_filesystem_paths(tmp_path) -> None:
    with pytest.raises(ValidationError) as info:
        EvidenceSnapshot.model_validate(
            {"kind": "paste", "manifest_sha256": "0" * 64, "pasted": {"query": ["SECRET-VALUE"]}}
        )
    assert "SECRET-VALUE" not in _error_detail(info.value)
    missing = FileNotFoundError(2, "No such file or directory", str(tmp_path / "secret.txt"))
    assert str(tmp_path) not in _error_detail(missing)
