import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import uvicorn

from demo_target.app import app
from eva_agent.domain.run import RunStatus
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.dependencies import build_runtime_dependencies
from eva_agent.runtime.manager import RunManager
from eva_agent.runtime.probe import ProbeStore
from tests.support.runtime import FakePlanner
from tests.support.target_runtime import runtime_config, target_plan, target_state


def start_demo_server() -> tuple[uvicorn.Server, threading.Thread]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8100, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            return server, thread
        time.sleep(0.05)
    raise RuntimeError("demo target did not start on 127.0.0.1:8100")


@pytest.fixture
def demo_server() -> Iterator[None]:
    server, thread = start_demo_server()
    try:
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=5)


async def test_target_run_against_real_http(tmp_path: Path, demo_server: None) -> None:
    config = runtime_config(tmp_path)
    dependencies = replace(
        build_runtime_dependencies(config, env={"FAKE_API_KEY": "test-key"}),
        planner=FakePlanner(plan=target_plan()),
    )
    state = target_state()

    async with RunManager.open(config, dependencies=dependencies) as manager:
        await manager.start(state["input"], state["budget"], run_id="run-1")
        view = await manager.approve("run-1", approved=True)
        report = await manager.report("run-1")

    assert view.status is RunStatus.COMPLETED
    assert report["probe"]["total"] == 2
    assert report["probe"]["succeeded"] == 2
    latency = next(item for item in report["dimensions"] if item["dimension"] == "latency")
    cost = next(item for item in report["dimensions"] if item["dimension"] == "cost")
    assert latency["metrics"]["count"] == 2
    assert latency["metrics"]["p95_ms"] >= 0
    assert cost["metrics"]["input_tokens_total"] > 0
    assert cost["metrics"]["output_tokens_total"] > 0
    assert report["metrics"]["overall_score"] == 1.0

    records = ProbeStore(run_paths(tmp_path / "data", "run-1").target_probe).read()
    assert [record.status_code for record in records] == [200, 200]
    assert all(record.usage_source == "target" for record in records)
