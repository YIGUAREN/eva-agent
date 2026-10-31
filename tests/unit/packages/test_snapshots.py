import pytest

from eva_agent.adapters.openjudge.cache import GradingCache, cache_key
from eva_agent.domain.grader import CheckSpec, GraderSpec, ScoreSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.result import GraderSampleOutcome
from eva_agent.domain.run import GraderSelection
from eva_agent.packages.manifest import PackageError
from eva_agent.packages.snapshots import collect_snapshots

ROWS = [{"query": f"q{index}", "response": f"a{index}"} for index in range(5)]


def _plan(
    *specs: GraderSpec,
    dimension: Dimension = Dimension.QUALITY,
    sample_size: int | None = None,
) -> DimensionPlan:
    return DimensionPlan(
        dimension=dimension,
        enabled=True,
        sample_size=sample_size,
        graders=list(specs),
    )


def _write(
    cache: GradingCache,
    spec: GraderSpec,
    rows: list[dict[str, str]],
    *,
    score: float | None = 0.75,
    error: str | None = None,
) -> None:
    for index, row in enumerate(rows):
        cache.write(
            spec.name,
            cache_key(spec, row),
            GraderSampleOutcome(
                grader=spec.name,
                sample_index=index,
                score=None if error is not None else score,
                error=error,
            ),
        )


def test_collect_snapshots_reads_every_recorded_outcome(tmp_path) -> None:
    spec = GraderSpec(level="builtin", name="correctness", builtin_id="correctness")
    rows = ROWS[:2]
    cache = GradingCache(tmp_path, "run-1")
    _write(cache, spec, rows)
    snapshots, blockers = collect_snapshots(
        data_dir=tmp_path,
        run_id="run-1",
        dimensions=[_plan(spec)],
        rows=rows,
        dimension_limits={"quality": 2},
    )
    assert [item.sample_index for item in snapshots.outcomes] == [0, 1]
    assert snapshots.outcomes[0].dimension is Dimension.QUALITY
    assert snapshots.for_spec(Dimension.QUALITY, "correctness")[0].score == 0.75
    assert blockers == []


def test_collect_snapshots_reports_a_missing_sample(tmp_path) -> None:
    spec = GraderSpec(level="builtin", name="correctness", builtin_id="correctness")
    rows = ROWS[:2]
    cache = GradingCache(tmp_path, "run-1")
    _write(cache, spec, rows[:1], score=0.5)
    _, blockers = collect_snapshots(
        data_dir=tmp_path,
        run_id="run-1",
        dimensions=[_plan(spec)],
        rows=rows,
        dimension_limits={"quality": 2},
    )
    assert blockers == ["quality/correctness: 1 of 2 samples were recorded"]


def test_declarative_specs_need_no_snapshot(tmp_path) -> None:
    spec = GraderSpec(
        level="declarative",
        name="cost-check",
        checks=[CheckSpec(name="budget", type="numeric_threshold", field="cost_usd", op="<=", value=1.0)],
        score=ScoreSpec(aggregation="weighted_mean", weights={"budget": 1.0}),
    )
    snapshots, blockers = collect_snapshots(
        data_dir=tmp_path,
        run_id="run-1",
        dimensions=[_plan(spec)],
        rows=ROWS[:1],
        dimension_limits={"quality": 1},
    )
    assert snapshots.outcomes == []
    assert blockers == []


def test_each_dimension_is_sliced_to_its_own_limit(tmp_path) -> None:
    "A shorter dimension must not be blamed for rows the longer dimension read."
    quality_spec = GraderSpec(level="builtin", name="correctness", builtin_id="correctness")
    quality_rows = ROWS[:2]
    security_rows = ROWS[:5]
    cache = GradingCache(tmp_path, "run-1")
    _write(cache, quality_spec, quality_rows)
    security_spec = GraderSpec(level="builtin", name="harmfulness", builtin_id="harmfulness")
    _write(cache, security_spec, security_rows)

    snapshots, blockers = collect_snapshots(
        data_dir=tmp_path,
        run_id="run-1",
        dimensions=[
            _plan(quality_spec, dimension=Dimension.QUALITY),
            _plan(dimension=Dimension.SECURITY),
        ],
        rows=ROWS,
        dimension_limits={"quality": 2, "security": 5},
    )
    assert blockers == []
    assert [item.sample_index for item in snapshots.for_spec(Dimension.QUALITY, "correctness")] == [0, 1]
    assert [item.sample_index for item in snapshots.for_spec(Dimension.SECURITY, "harmfulness")] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_security_snapshots_follow_the_grader_selection(tmp_path) -> None:
    code_rows = [{"query": "q0", "response": "def f():\n    return 1"}]
    cache = GradingCache(tmp_path, "run-1")
    for name in ("harmfulness", "code_security"):
        _write(cache, GraderSpec(level="builtin", name=name, builtin_id=name), code_rows)

    snapshots, blockers = collect_snapshots(
        data_dir=tmp_path,
        run_id="run-1",
        dimensions=[_plan(dimension=Dimension.SECURITY)],
        rows=code_rows,
        dimension_limits={"security": 1},
        graders=GraderSelection(security=["code_security"]),
    )

    assert blockers == []
    assert [item.spec_name for item in snapshots.outcomes] == ["code_security"]


def test_same_named_specs_in_different_dimensions_do_not_merge(tmp_path) -> None:
    spec = GraderSpec(level="builtin", name="correctness", builtin_id="correctness")
    cache = GradingCache(tmp_path, "run-1")
    _write(cache, spec, ROWS[:2])
    snapshots, blockers = collect_snapshots(
        data_dir=tmp_path,
        run_id="run-1",
        dimensions=[
            _plan(spec, dimension=Dimension.QUALITY),
            _plan(spec, dimension=Dimension.LATENCY),
        ],
        rows=ROWS[:2],
        dimension_limits={"quality": 2, "latency": 2},
    )
    assert blockers == []
    assert len(snapshots.outcomes) == 4
    assert len(snapshots.for_spec(Dimension.QUALITY, "correctness")) == 2
    assert len(snapshots.for_spec(Dimension.LATENCY, "correctness")) == 2


def test_errored_recorded_outcomes_are_a_blocker(tmp_path) -> None:
    spec = GraderSpec(level="builtin", name="correctness", builtin_id="correctness")
    rows = ROWS[:2]
    cache = GradingCache(tmp_path, "run-1")
    _write(cache, spec, rows, error="grader unavailable")
    snapshots, blockers = collect_snapshots(
        data_dir=tmp_path,
        run_id="run-1",
        dimensions=[_plan(spec)],
        rows=rows,
        dimension_limits={"quality": 2},
    )
    assert len(snapshots.outcomes) == 2
    assert blockers == ["quality/correctness: 2 of 2 recorded samples errored"]


def test_missing_dimension_limit_fails_closed(tmp_path) -> None:
    spec = GraderSpec(level="builtin", name="correctness", builtin_id="correctness")
    with pytest.raises(PackageError, match="sample limit"):
        collect_snapshots(
            data_dir=tmp_path,
            run_id="run-1",
            dimensions=[_plan(spec)],
            rows=ROWS[:1],
            dimension_limits={},
        )
