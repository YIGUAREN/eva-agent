import json
from decimal import Decimal
from typing import Any

import pytest

from eva_agent.adapters.openjudge.cache import GradingCache, cache_key
from eva_agent.adapters.target import tokens as tokens_module
from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.config.settings import ModelConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.pricing import ModelPrice, PricingTable
from eva_agent.tools.graders import RunGradersInput, build_run_graders_tool


class FakeEncoding:
    """Offline stand-in for ``tiktoken.Encoding``: one token per character."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch: pytest.MonkeyPatch):
    tokens_module._encoding.cache_clear()
    monkeypatch.setattr(tokens_module.tiktoken, "get_encoding", lambda name: FakeEncoding())
    yield
    tokens_module._encoding.cache_clear()


class NoopStore:
    """A limited ledger requires a store; these tests do not persist."""

    async def load(self, run_id: str) -> None:
        return None

    async def save(self, run_id: str, usage: Any) -> None:
        return None


def model() -> ModelConfig:
    return ModelConfig(provider="anthropic", model="demo-echo", api_key_env="DEMO_KEY")


def spec(rubric: str = "- clear") -> GraderSpec:
    return GraderSpec(level="llm_rubric", name="quality", rubric=rubric)


def row(response: str) -> dict[str, Any]:
    return {"query": "q", "response": response}


class RecordingRunner:
    def __init__(self) -> None:
        self.calls = 0
        self.rows: list[list[dict[str, Any]]] = []

    async def run(self, specs: list[GraderSpec], rows: list[dict[str, Any]]) -> GraderRunOutput:
        self.calls += 1
        self.rows.append(rows)
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(grader=item.name, sample_index=index, score=0.75)
                for item in specs
                for index in range(len(rows))
            ],
            duration_ms=1,
        )


def payload(*, rubric: str = "- clear", rows: list[dict[str, Any]] | None = None) -> RunGradersInput:
    return RunGradersInput(
        specs=[spec(rubric)],
        rows=rows if rows is not None else [row("a"), row("b")],
        run_id="run-1",
    )


async def test_a_second_call_serves_every_sample_from_cache(tmp_path) -> None:
    runner = RecordingRunner()
    tool = build_run_graders_tool(runner, max_rows=10, data_dir=str(tmp_path))
    assert tool.handler is not None
    first = await tool.handler(payload())
    second = await tool.handler(payload())
    assert (first.cached, first.fresh) == (0, 2)
    assert (second.cached, second.fresh) == (2, 0)
    assert runner.calls == 1
    assert [item.sample_index for item in second.outcomes] == [0, 1]
    assert all(item.score == 0.75 for item in second.outcomes)


async def test_only_changed_rows_are_fresh_and_indexes_are_remapped(tmp_path) -> None:
    runner = RecordingRunner()
    tool = build_run_graders_tool(runner, max_rows=10, data_dir=str(tmp_path))
    assert tool.handler is not None
    await tool.handler(payload())
    output = await tool.handler(payload(rows=[row("a"), row("c")]))
    assert (output.cached, output.fresh) == (1, 1)
    assert runner.rows[-1] == [row("c")]
    assert [item.sample_index for item in output.outcomes] == [0, 1]


async def test_a_changed_grader_spec_hashes_to_a_new_key(tmp_path) -> None:
    runner = RecordingRunner()
    tool = build_run_graders_tool(runner, max_rows=10, data_dir=str(tmp_path))
    assert tool.handler is not None
    await tool.handler(payload())
    output = await tool.handler(payload(rubric="- different"))
    assert (output.cached, output.fresh) == (0, 2)
    assert runner.calls == 2


async def test_budget_exhaustion_keeps_earlier_outcomes_and_cached_hits(tmp_path) -> None:
    runner = RecordingRunner()
    tool = build_run_graders_tool(runner, max_rows=10, judge_model=model(), data_dir=str(tmp_path))
    assert tool.handler is not None
    spec_a = GraderSpec(level="llm_rubric", name="alpha", rubric="- a")
    spec_b = GraderSpec(level="llm_rubric", name="beta", rubric="- b")
    # Warm the cache: both graders for row "a", plus alpha for row "b".
    await tool.handler(RunGradersInput(specs=[spec_a, spec_b], rows=[row("a")], run_id="run-1"))
    await tool.handler(RunGradersInput(specs=[spec_a], rows=[row("b")], run_id="run-1"))
    calls_before = runner.calls

    estimator = CostEstimator(
        PricingTable(
            [
                ModelPrice(
                    provider="anthropic",
                    model="demo-echo",
                    input_per_million=Decimal("0"),
                    output_per_million=Decimal("1"),
                )
            ]
        )
    )
    ledger = BudgetLedger(
        Budget(max_cost_usd=0.000001, max_wall_clock_s=60.0, max_target_requests=10),
        store=NoopStore(),
    )
    with bind_budget(ledger, estimator, BudgetModels(agent=model(), judge=model(), attack=model())):
        output = await tool.handler(
            RunGradersInput(
                specs=[spec_a, spec_b],
                rows=[row("a"), row("b")],
                run_id="run-1",
            )
        )

    # Spec alpha is fully cached, so spec beta's fresh row is the one refused;
    # the refusal must still return alpha's earlier outcomes plus beta's hit.
    assert output.truncated_reason == "budget"
    assert [(item.grader, item.sample_index) for item in output.outcomes] == [
        ("alpha", 0),
        ("alpha", 1),
        ("beta", 0),
    ]
    assert (output.cached, output.fresh) == (3, 0)
    assert runner.calls == calls_before
    assert ledger.snapshot().reserved_usd == Decimal("0")


class ErrorRunner:
    """Every sample errors, so nothing should be cached for a retry."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, specs: list[GraderSpec], rows: list[dict[str, Any]]) -> GraderRunOutput:
        self.calls += 1
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(grader=item.name, sample_index=index, error="grader unavailable")
                for item in specs
                for index in range(len(rows))
            ],
            duration_ms=1,
        )


def _write_record(cache: GradingCache, grader: str, key: str, data: dict[str, Any]) -> None:
    path = cache._path(grader, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_structurally_invalid_record_is_a_miss(tmp_path) -> None:
    cache = GradingCache(tmp_path, "run-1")
    key = "a" * 64
    _write_record(cache, "quality", key, {"cache_key": key, "score": 7.5})

    assert cache.read("quality", key, 0) is None


def test_a_record_without_score_or_error_is_a_miss(tmp_path) -> None:
    cache = GradingCache(tmp_path, "run-1")
    key = "b" * 64
    _write_record(cache, "quality", key, {"cache_key": key})

    assert cache.read("quality", key, 0) is None


async def test_an_invalid_cache_record_is_re_graded(tmp_path) -> None:
    runner = RecordingRunner()
    tool = build_run_graders_tool(runner, max_rows=10, data_dir=str(tmp_path))
    assert tool.handler is not None
    one_row = payload(rows=[row("a")])
    await tool.handler(one_row)

    key = cache_key(spec(), row("a"))
    path = GradingCache(tmp_path, "run-1")._path("quality", key)
    path.write_text(json.dumps({"cache_key": key, "score": 7.5}), encoding="utf-8")

    output = await tool.handler(one_row)
    assert (output.cached, output.fresh) == (0, 1)
    assert output.outcomes[0].score == 0.75
    assert runner.calls == 2


async def test_an_error_outcome_is_not_cached_and_is_regraded(tmp_path) -> None:
    runner = ErrorRunner()
    tool = build_run_graders_tool(runner, max_rows=10, data_dir=str(tmp_path))
    assert tool.handler is not None
    first = await tool.handler(payload())
    second = await tool.handler(payload())

    assert (first.cached, first.fresh) == (0, 2)
    assert (second.cached, second.fresh) == (0, 2)
    assert runner.calls == 2


async def test_a_cache_write_failure_does_not_fail_the_grading_run(tmp_path) -> None:
    runner = RecordingRunner()
    tool = build_run_graders_tool(runner, max_rows=10, data_dir=str(tmp_path))
    assert tool.handler is not None
    # A file where the cache wants a directory makes mkdir/write raise OSError.
    conflict = tmp_path / "runs" / "run-1" / "gradings" / "quality"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("not a directory", encoding="utf-8")

    first = await tool.handler(payload())
    second = await tool.handler(payload())

    assert (first.cached, first.fresh) == (0, 2)
    assert [item.score for item in first.outcomes] == [0.75, 0.75]
    # Nothing was cached, so the retry grades again instead of failing.
    assert (second.cached, second.fresh) == (0, 2)
    assert runner.calls == 2


def test_a_windows_reserved_grader_name_is_safe(tmp_path) -> None:
    cache = GradingCache(tmp_path, "run-1")
    key = "c" * 64

    cache.write("CON", key, GraderSampleOutcome(grader="CON", sample_index=0, score=0.5))

    assert cache._path("CON", key).parent.name != "CON"
    hit = cache.read("CON", key, 0)
    assert hit is not None and hit.score == 0.5


async def test_cached_rows_do_not_charge_the_ledger_again(tmp_path) -> None:
    runner = RecordingRunner()
    context = BudgetModels(agent=model(), judge=model(), attack=model())
    estimator = CostEstimator(
        PricingTable(
            [
                ModelPrice(
                    provider="anthropic",
                    model="demo-echo",
                    input_per_million=Decimal("0"),
                    output_per_million=Decimal("1"),
                )
            ]
        )
    )
    tool = build_run_graders_tool(
        runner,
        max_rows=10,
        judge_model=model(),
        data_dir=str(tmp_path),
    )
    ledger = BudgetLedger(
        Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
        store=NoopStore(),
    )
    with bind_budget(ledger, estimator, context):
        assert tool.handler is not None
        await tool.handler(payload())
        spent_after_first = ledger.snapshot().spent_usd
        assert spent_after_first > Decimal("0")
        await tool.handler(payload())
        assert ledger.snapshot().spent_usd == spent_after_first
        assert ledger.snapshot().reserved_usd == Decimal("0")
