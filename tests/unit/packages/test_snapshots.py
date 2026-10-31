from eva_agent.adapters.openjudge.cache import GradingCache, cache_key
from eva_agent.domain.grader import CheckSpec, GraderSpec, ScoreSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.result import GraderSampleOutcome
from eva_agent.packages.snapshots import collect_snapshots


def _plan(*specs: GraderSpec) -> DimensionPlan:
    return DimensionPlan(dimension=Dimension.QUALITY, enabled=True, graders=list(specs))


def test_collect_snapshots_reads_every_recorded_outcome(tmp_path) -> None:
    spec = GraderSpec(level="builtin", name="correctness", builtin_id="correctness")
    rows = [{"query": "q1", "response": "a1"}, {"query": "q2", "response": "a2"}]
    cache = GradingCache(tmp_path, "run-1")
    for index, row in enumerate(rows):
        cache.write(
            spec.name,
            cache_key(spec, row),
            GraderSampleOutcome(grader=spec.name, sample_index=index, score=0.75, raw_score=3.0),
        )
    snapshots, blockers = collect_snapshots(
        data_dir=tmp_path, run_id="run-1", dimensions=[_plan(spec)], rows=rows
    )
    assert [item.sample_index for item in snapshots.outcomes] == [0, 1]
    assert snapshots.for_spec("correctness")[0].score == 0.75
    assert blockers == []


def test_collect_snapshots_reports_a_missing_sample(tmp_path) -> None:
    spec = GraderSpec(level="builtin", name="correctness", builtin_id="correctness")
    rows = [{"query": "q1", "response": "a1"}, {"query": "q2", "response": "a2"}]
    cache = GradingCache(tmp_path, "run-1")
    cache.write(
        spec.name,
        cache_key(spec, rows[0]),
        GraderSampleOutcome(grader=spec.name, sample_index=0, score=0.5),
    )
    _, blockers = collect_snapshots(data_dir=tmp_path, run_id="run-1", dimensions=[_plan(spec)], rows=rows)
    assert blockers == ["quality/correctness: 1 of 2 samples were recorded"]


def test_declarative_specs_need_no_snapshot(tmp_path) -> None:
    spec = GraderSpec(
        level="declarative",
        name="cost-check",
        checks=[CheckSpec(name="budget", type="numeric_threshold", field="cost_usd", op="<=", value=1.0)],
        score=ScoreSpec(aggregation="weighted_mean", weights={"budget": 1.0}),
    )
    snapshots, blockers = collect_snapshots(
        data_dir=tmp_path, run_id="run-1", dimensions=[_plan(spec)], rows=[{"query": "q", "response": "a"}]
    )
    assert snapshots.outcomes == []
    assert blockers == []
