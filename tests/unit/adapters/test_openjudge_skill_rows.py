import pytest
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderScore

from eva_agent.adapters.openjudge import runner as runner_module
from eva_agent.adapters.openjudge.catalog import GraderEntry
from eva_agent.adapters.openjudge.runner import OpenJudgeRunner
from eva_agent.domain.grader import GraderSpec


class SkillShapeGrader(BaseGrader):
    def __init__(self) -> None:
        super().__init__(name="threat_analysis")

    async def _aevaluate(self, **kwargs):
        assert isinstance(kwargs["script_contents"], list)
        assert isinstance(kwargs["reference_contents"], list)
        return GraderScore(name=self.name, reason="counted", score=float(len(kwargs["script_contents"])))


def catalog() -> dict[str, GraderEntry]:
    return {
        "threat_analysis": GraderEntry(
            id="threat_analysis",
            klass="ignored.package.SkillShapeGrader",  # pyright: ignore[reportCallIssue]
            mode="pointwise",
            inputs=[
                "skill_name",
                "skill_manifest",
                "instruction_body",
                "script_contents",
                "reference_contents",
            ],
            tags=["security", "skill"],
            needs_model=False,
            score_min=1,
            score_max=4,
        )
    }


@pytest.mark.asyncio
async def test_list_valued_row_inputs_reach_the_grader(monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "load_class", lambda path: SkillShapeGrader)
    runner = OpenJudgeRunner(object(), catalog(), max_concurrency=2, timeout_s=5.0)  # pyright: ignore[reportArgumentType]
    row = {
        "skill_name": "greeter",
        "skill_manifest": "name: greeter\n",
        "instruction_body": "# Greeter\n",
        "script_contents": ["print('hi')", "print('bye')"],
        "reference_contents": [],
    }
    output = await runner.run(
        [GraderSpec(level="builtin", name="threat_analysis", builtin_id="threat_analysis")],
        [row],
    )
    assert output.error is None
    assert output.outcomes[0].error is None
    assert output.outcomes[0].raw_score == 2.0
    assert output.outcomes[0].score == pytest.approx(1 / 3)
