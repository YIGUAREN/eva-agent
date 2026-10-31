from eva_agent.domain.plan import Dimension
from eva_agent.domain.result import DimensionResult, MergedResult, Usage
from eva_agent.domain.run import RunStatus
from eva_agent.graph.nodes import report as report_module
from eva_agent.graph.nodes.report import report_node
from tests.support.runtime import dependencies, initial_state, paste_input, sample_plan


def _merged() -> MergedResult:
    return MergedResult(
        status=RunStatus.COMPLETED,
        results=[
            DimensionResult(
                dimension=Dimension.QUALITY,
                status="ok",
                score=1.0,
                confidence=1.0,
                summary="ok",
                usage=Usage(),
            )
        ],
        skipped=[],
        metrics={},
    )


async def test_malformed_budget_usage_fails_the_report_event(tmp_path, monkeypatch) -> None:
    events: list[tuple[str, dict]] = []

    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append((event_type, payload or {}))

    def boom():
        raise TypeError("malformed usage")

    monkeypatch.setattr(report_module, "budget_context", boom)
    deps = dependencies(tmp_path, event_sink=sink)
    state = initial_state(paste_input())
    state["plan"] = sample_plan()
    state["merged"] = _merged()

    result = await report_node(state, deps)

    assert result["status"] is RunStatus.FAILED
    assert result["errors"] == ["生成报告失败: malformed usage"]
    assert events == [("report.failed", {"error": "malformed usage"})]
