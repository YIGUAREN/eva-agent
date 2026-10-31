import pytest

from eva_agent.adapters.openjudge.catalog import GraderEntry
from eva_agent.tools.graders import ListGradersInput, build_list_graders_tool


def entry(grader_id: str, tags: list[str]) -> GraderEntry:
    return GraderEntry(
        id=grader_id,
        klass="openjudge.graders.common.relevance.RelevanceGrader",  # pyright: ignore[reportCallIssue]
        mode="pointwise",
        inputs=["query", "response"],
        tags=tags,
        needs_model=True,
        score_min=1,
        score_max=5,
    )


def catalog() -> dict[str, GraderEntry]:
    return {
        "relevance": entry("relevance", ["quality"]),
        "threat_analysis": entry("threat_analysis", ["security", "skill"]),
    }


@pytest.mark.asyncio
async def test_default_listing_hides_package_level_skill_graders() -> None:
    tool = build_list_graders_tool(catalog())
    output = await tool.handler(ListGradersInput())
    assert [item["id"] for item in output.graders] == ["relevance"]


@pytest.mark.asyncio
async def test_tag_filter_still_matches_skill_graders() -> None:
    tool = build_list_graders_tool(catalog())
    output = await tool.handler(ListGradersInput(tags=["skill"]))
    assert [item["id"] for item in output.graders] == ["threat_analysis"]
