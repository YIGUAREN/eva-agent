from pathlib import Path
from typing import cast

from eva_agent.domain.run import RunStatus
from eva_agent.graph.nodes.latency import latency_node
from eva_agent.graph.state import ExpertState
from tests.support.runtime import budget, dependencies


async def test_missing_evidence_emits_latency_failed(tmp_path: Path) -> None:
    events: list[str] = []

    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append(event_type)

    deps = dependencies(tmp_path, event_sink=sink)
    state = cast(
        ExpertState,
        {
            "run_id": "run-1",
            "dimension_plan": None,
            "evidence": None,
            "probe": None,
            "budget": budget(),
        },
    )

    result = await latency_node(state, deps)

    assert result["status"] is RunStatus.FAILED
    assert result["errors"] == ["latency 专家缺少计划或证据"]
    assert events == ["latency.failed"]
