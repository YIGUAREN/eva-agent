import pytest

from eva_agent.adapters.openjudge.catalog import load_catalog
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.result import GraderRunOutput, GraderSampleOutcome
from eva_agent.tools.bundle import ToolDependencies, build_tool_registry
from eva_agent.tools.graders import ListGradersInput, RunGradersInput
from eva_agent.tools.rubric import GenerateRubricInput


class FakeRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, specs: list[GraderSpec], rows: list[dict[str, str]]) -> GraderRunOutput:
        self.calls += 1
        return GraderRunOutput(
            outcomes=[
                GraderSampleOutcome(grader=spec.name, sample_index=index, score=0.5)
                for spec in specs
                for index in range(len(rows))
            ],
            duration_ms=10,
        )


class FakeRubricGenerator:
    async def generate(
        self,
        *,
        task_description: str,
        sample_queries: list[str],
        scenario: str | None = None,
    ) -> list[str]:
        assert task_description
        return ["回答准确", "回答完整"]


def build_registry() -> tuple:
    catalog = load_catalog(__import__("pathlib").Path("config/graders.yaml"))
    runner = FakeRunner()
    generator = FakeRubricGenerator()
    registry = build_tool_registry(
        ToolDependencies(
            catalog=catalog,
            runner=runner,
            rubric_generator=generator,
            max_bytes=10_000,
            max_rows=10,
        )
    )
    return registry, runner


async def test_registry_contains_phase_one_tools() -> None:
    registry, _ = build_registry()

    assert registry.names() == [
        "count_tokens",
        "generate_rubric",
        "list_graders",
        "read_evidence",
        "read_target_probe",
        "run_declarative_checks",
        "run_graders",
        "sample_dataset",
    ]


async def test_list_graders_filters_by_tag() -> None:
    registry, _ = build_registry()

    result = await registry.call("list_graders", ListGradersInput(tags=["reference"]))

    assert {item["id"] for item in result.graders} == {"correctness", "similarity"}


async def test_list_graders_defaults_to_all() -> None:
    registry, _ = build_registry()

    result = await registry.call("list_graders", ListGradersInput())

    assert len(result.graders) == 5


async def test_run_graders_delegates_to_runner() -> None:
    registry, runner = build_registry()
    spec = GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")

    result = await registry.call(
        "run_graders",
        RunGradersInput(specs=[spec], rows=[{"query": "q", "response": "a", "reference": "r", "context": ""}]),
    )

    assert runner.calls == 1
    assert result.outcomes[0].score == 0.5


async def test_generate_rubric_returns_strings() -> None:
    registry, _ = build_registry()

    result = await registry.call(
        "generate_rubric",
        GenerateRubricInput(task_description="评估回答质量", sample_queries=["q"]),
    )

    assert result.rubrics == ["回答准确", "回答完整"]


async def test_run_graders_rejects_too_many_rows() -> None:
    registry, runner = build_registry()
    spec = GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")
    rows = [{"query": f"q{index}", "response": "a", "reference": "r", "context": ""} for index in range(11)]

    with pytest.raises(ValueError, match="max_rows"):
        await registry.call("run_graders", RunGradersInput(specs=[spec], rows=rows))

    assert runner.calls == 0
