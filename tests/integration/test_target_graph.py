from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from eva_agent.domain.plan import Dimension
from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.graph.build import build_graph
from eva_agent.runtime.artifacts import run_paths
from tests.support.target_runtime import target_dependencies, target_state

CONFIG = {"configurable": {"thread_id": "run-1"}}


async def test_target_run_probes_once_then_fans_out(tmp_path: Path) -> None:
    calls: list[str] = []
    events: list[str] = []
    app = build_graph(target_dependencies(tmp_path, calls, events), MemorySaver())

    result = await app.ainvoke(target_state(), CONFIG)

    assert result["status"] is RunStatus.WAITING_APPROVAL
    assert calls == []
    assert result["probe"] is None
    assert result["evidence"] is not None
    assert result["evidence"].kind is InputKind.TARGET

    await app.ainvoke(Command(resume={"approved": True}), CONFIG)
    snapshot = await app.aget_state(CONFIG)

    assert snapshot.values["status"] is RunStatus.COMPLETED
    assert calls == ["a", "b"]
    assert snapshot.values["probe"].total == 2
    assert snapshot.values["probe"].succeeded == 2
    dimensions = {item.dimension for item in snapshot.values["expert_results"]}
    assert dimensions == {Dimension.LATENCY, Dimension.COST}
    assert all(item.status == "ok" for item in snapshot.values["expert_results"])
    assert snapshot.values["merged"].metrics["overall_score"] == 1.0

    paths = run_paths(tmp_path / "data", "run-1")
    assert paths.target_probe.is_file()
    assert paths.report_json.is_file()
    assert events.index("probe.finished") < events.index("dispatch.ready")
    assert "latency.completed" in events
    assert "cost.completed" in events
