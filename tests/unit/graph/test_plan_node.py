from pathlib import Path
from typing import cast

from eva_agent.domain.run import RunStatus
from eva_agent.graph.nodes.plan import plan_node
from eva_agent.graph.state import EvalState
from tests.support.runtime import dependencies, initial_state, paste_input


async def test_missing_evidence_fails_without_calling_planner(tmp_path: Path) -> None:
    events: list[str] = []

    async def sink(run_id: str, event_type: str, payload: dict | None = None) -> None:
        events.append(event_type)

    deps = dependencies(tmp_path, event_sink=sink)
    state = cast(EvalState, {**initial_state(paste_input()), "evidence": None})

    result = await plan_node(state, deps)

    assert result["status"] is RunStatus.FAILED
    assert result["errors"] == ["缺少证据，无法制定计划"]
    assert "plan.failed" in events
