from langgraph.checkpoint.memory import MemorySaver

from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind, PastedInput, RunInput, RunStatus
from eva_agent.graph.build import build_graph
from eva_agent.graph.planner import StubPlanner
from eva_agent.graph.state import EvalState


def initial_state(kind: InputKind = InputKind.PASTE, **input_kwargs: object) -> EvalState:
    return {
        "run_id": "run-1",
        "input": RunInput(kind=kind, **input_kwargs),  # type: ignore[arg-type]
        "budget": Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=10),
        "evidence": None,
        "plan": None,
        "status": RunStatus.QUEUED,
        "errors": [],
    }


async def test_paste_run_reaches_completed() -> None:
    app = build_graph(StubPlanner(), MemorySaver())
    config = {"configurable": {"thread_id": "run-1"}}

    result = await app.ainvoke(initial_state(paste=PastedInput(query="q", response="a")), config)

    assert result["status"] is RunStatus.COMPLETED
    assert result["evidence"] is not None
    assert len(result["evidence"].manifest_sha256) == 64
    assert result["plan"] is not None
    assert result["plan"].objective
    assert len(result["plan"].skipped) == 4


async def test_checkpoint_is_written() -> None:
    app = build_graph(StubPlanner(), MemorySaver())
    config = {"configurable": {"thread_id": "run-2"}}

    await app.ainvoke(initial_state(paste=PastedInput(query="q")), config)
    snapshot = await app.aget_state(config)

    assert snapshot.values["status"] is RunStatus.COMPLETED


async def test_unsupported_input_fails_without_raising() -> None:
    app = build_graph(StubPlanner())

    result = await app.ainvoke(initial_state(kind=InputKind.UPLOAD, upload_id="upload-1"))

    assert result["status"] is RunStatus.FAILED
    assert result["errors"]
    assert result["plan"] is None
