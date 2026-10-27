import asyncio

import pytest
from openjudge.graders.base_grader import BaseGrader, GraderMode
from openjudge.graders.schema import GraderError, GraderScore

from eva_agent.adapters.openjudge.catalog import load_catalog
from eva_agent.adapters.openjudge.runner import OpenJudgeRunner, _TimeoutGrader
from eva_agent.domain.grader import GraderSpec
from tests.fakes.openjudge import FakeChatModel

CATALOG = load_catalog(__import__("pathlib").Path("config/graders.yaml"))


class SlowGrader(BaseGrader):
    def __init__(self) -> None:
        super().__init__(name="slow", mode=GraderMode.POINTWISE)

    async def _aevaluate(self, **kwargs) -> GraderScore:
        await asyncio.sleep(0.05)
        return GraderScore(name="slow", score=1, reason="slow")


async def test_builtin_grader_runs_against_fake_model() -> None:
    runner = OpenJudgeRunner(
        FakeChatModel({"reason": "great", "score": 5}),
        CATALOG,
        max_concurrency=2,
        timeout_s=5.0,
    )
    spec = GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")

    output = await runner.run(
        [spec],
        [{"query": "q", "response": "a", "reference": "a", "context": ""}],
    )

    assert output.error is None
    assert output.outcomes[0].score == 1.0
    assert output.outcomes[0].raw_score == 5.0
    assert output.outcomes[0].error is None


async def test_llm_rubric_grader_runs_against_fake_model() -> None:
    runner = OpenJudgeRunner(
        FakeChatModel({"reason": "helpful", "score": 1}),
        CATALOG,
        max_concurrency=1,
        timeout_s=5.0,
    )
    spec = GraderSpec(level="llm_rubric", name="helpfulness", mode="pointwise", rubric="回答必须有用")

    output = await runner.run([spec], [{"query": "q", "response": "a", "reference": "", "context": ""}])

    assert output.outcomes[0].score == 1.0
    assert output.outcomes[0].error is None


async def test_unknown_builtin_id_is_isolated() -> None:
    runner = OpenJudgeRunner(FakeChatModel(), CATALOG, max_concurrency=1, timeout_s=1.0)
    spec = GraderSpec(level="builtin", name="missing", mode="pointwise", builtin_id="missing")

    output = await runner.run([spec], [{"query": "q", "response": "a", "reference": "", "context": ""}])

    assert output.outcomes[0].score is None
    assert "unknown grader" in (output.outcomes[0].error or "")


async def test_whole_runner_failure_is_returned_as_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExplodingRunner:
        def __init__(self, **kwargs):
            pass

        async def arun(self, rows):
            raise RuntimeError("runner exploded")

    monkeypatch.setattr("eva_agent.adapters.openjudge.runner.GradingRunner", ExplodingRunner)
    runner = OpenJudgeRunner(FakeChatModel(), CATALOG, max_concurrency=1, timeout_s=1.0)
    spec = GraderSpec(level="builtin", name="correctness", mode="pointwise", builtin_id="correctness")

    output = await runner.run([spec], [{"query": "q", "response": "a", "reference": "a", "context": ""}])

    assert output.error is not None
    assert output.outcomes[0].error is not None


async def test_timeout_grader_raises_timeout_error() -> None:
    grader = _TimeoutGrader(SlowGrader(), timeout_s=0.01)

    with pytest.raises(TimeoutError, match="timed out"):
        await grader.aevaluate(query="q")


def test_grader_error_is_valid_openjudge_result() -> None:
    error = GraderError(name="x", error="boom", reason="bad")

    assert error.error == "boom"
