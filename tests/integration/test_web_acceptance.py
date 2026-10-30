from collections.abc import Iterator
from pathlib import Path

from httpx import AsyncClient

from eva_agent.runtime.manager import RunManager
from eva_agent.server.state import AppState
from eva_agent.worker.runner import RunWorker
from tests.support.api import PASTE_PAYLOAD, csrf_headers, login
from tests.support.runtime import dependencies


def _iter_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _assert_no_path(payload: object, tmp_path: Path) -> None:
    needle = str(tmp_path)
    for text in _iter_strings(payload):
        assert needle not in text


async def test_paste_run_end_to_end(client: AsyncClient, state: AppState, tmp_path: Path) -> None:
    csrf = await login(client)
    created = await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))
    assert created.status_code == 201
    run_id = created.json()["run_id"]

    manager = RunManager(state.config, dependencies=dependencies(tmp_path))
    await manager.setup()
    try:
        worker = RunWorker.from_manager(
            manager, data_dir=state.config.storage.data_dir, poll_interval_s=0.01
        )

        assert await worker.tick() is True
        waiting = await client.get(f"/api/runs/{run_id}")
        assert waiting.json()["status"] == "waiting_approval"

        approved = await client.post(
            f"/api/runs/{run_id}/approve",
            json={"approved": True},
            headers=csrf_headers(csrf),
        )
        assert approved.status_code == 202
        assert await worker.tick() is True

        completed = await client.get(f"/api/runs/{run_id}")
        assert completed.json()["status"] == "completed"
        report = await client.get(f"/api/runs/{run_id}/report")
        assert report.status_code == 200
        assert report.json()["report"]["run_id"] == run_id

        published = await client.post(
            f"/api/runs/{run_id}/publish",
            json={"published": True},
            headers=csrf_headers(csrf),
        )
        assert published.status_code == 200
        public_report = await client.get(f"/api/public/runs/{run_id}/report")
        assert public_report.status_code == 200
        assert str(tmp_path) not in public_report.text
        _assert_no_path(public_report.json(), tmp_path)
    finally:
        await manager.close()
