import io
import zipfile

import httpx
import pytest

from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import Dimension, EvalPlan, SkippedDimension
from eva_agent.runtime.manager import RunManager
from eva_agent.server.state import AppState
from eva_agent.worker.runner import RunWorker
from tests.support.api import csrf_headers, login
from tests.support.runtime import FakePlanner, dependencies


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _all_skipped_plan() -> EvalPlan:
    return EvalPlan(
        objective="评估上传证据",
        evidence_summary="上传证据没有 qa.jsonl",
        dimensions=[],
        skipped=[SkippedDimension(dimension=dimension, reason="证据没有可用响应") for dimension in Dimension],
        budget=Budget(max_cost_usd=1.0, max_wall_clock_s=60.0, max_target_requests=1),
    )


async def test_upload_run_end_to_end(client: httpx.AsyncClient, state: AppState, tmp_path) -> None:
    csrf = await login(client)
    uploaded = await client.post(
        "/api/uploads",
        files={
            "file": (
                "evidence.zip",
                _zip(
                    {
                        "qa.jsonl": b'{"query": "what is 2+2?", "response": "4"}\n',
                        "qa.txt": b"what is 2+2?\\n4",
                    }
                ),
                "application/zip",
            )
        },
        headers=csrf_headers(csrf),
    )
    assert uploaded.status_code == 202, uploaded.text
    upload_id = uploaded.json()["upload_id"]

    created = await client.post(
        "/api/runs",
        json={
            "input": {"kind": "upload", "upload_id": upload_id, "objective": "评估上传证据"},
            "budget": {"max_cost_usd": 1.0, "max_wall_clock_s": 60.0, "max_target_requests": 1},
        },
        headers=csrf_headers(csrf),
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["run_id"]

    manager = RunManager(state.config, dependencies=dependencies(tmp_path))
    await manager.setup()
    try:
        worker = RunWorker.from_manager(
            manager, data_dir=state.config.storage.data_dir, poll_interval_s=0.01
        )
        assert await worker.tick() is True
        waiting = await client.get(f"/api/runs/{run_id}")
        assert waiting.json()["status"] == "waiting_approval", waiting.text

        approved = await client.post(
            f"/api/runs/{run_id}/approve",
            json={"approved": True},
            headers=csrf_headers(csrf),
        )
        assert approved.status_code == 202
        assert await worker.tick() is True

        completed = await client.get(f"/api/runs/{run_id}")
        assert completed.json()["status"] == "completed", completed.text
        report = await client.get(f"/api/runs/{run_id}/report")
        assert report.status_code == 200
    finally:
        await manager.close()


async def test_upload_without_qa_completes_with_quality_skipped(
    client: httpx.AsyncClient, state: AppState, tmp_path
) -> None:
    csrf = await login(client)
    uploaded = await client.post(
        "/api/uploads",
        files={"file": ("evidence.zip", _zip({"qa.txt": b"what is 2+2?\\n4"}), "application/zip")},
        headers=csrf_headers(csrf),
    )
    assert uploaded.status_code == 202, uploaded.text
    upload_id = uploaded.json()["upload_id"]

    created = await client.post(
        "/api/runs",
        json={
            "input": {"kind": "upload", "upload_id": upload_id, "objective": "评估上传证据"},
            "budget": {"max_cost_usd": 1.0, "max_wall_clock_s": 60.0, "max_target_requests": 1},
        },
        headers=csrf_headers(csrf),
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["run_id"]

    manager = RunManager(
        state.config,
        dependencies=dependencies(tmp_path, planner=FakePlanner(plan=_all_skipped_plan())),
    )
    await manager.setup()
    try:
        worker = RunWorker.from_manager(
            manager, data_dir=state.config.storage.data_dir, poll_interval_s=0.01
        )
        assert await worker.tick() is True
        waiting = await client.get(f"/api/runs/{run_id}")
        assert waiting.json()["status"] == "waiting_approval", waiting.text

        approved = await client.post(
            f"/api/runs/{run_id}/approve",
            json={"approved": True},
            headers=csrf_headers(csrf),
        )
        assert approved.status_code == 202
        assert await worker.tick() is True

        completed = await client.get(f"/api/runs/{run_id}")
        assert completed.json()["status"] == "completed", completed.text
        report = await client.get(f"/api/runs/{run_id}/report")
        assert report.status_code == 200
        payload = report.json()["report"]
        assert payload["status"] == "completed"
        assert "quality" in {item["dimension"] for item in payload["skipped"]}
    finally:
        await manager.close()


@pytest.mark.parametrize(
    "members",
    [
        {"../evil.txt": b"x"},
        {"/absolute.txt": b"x"},
    ],
)
async def test_unsafe_upload_is_rejected_without_files(
    client: httpx.AsyncClient, state: AppState, members: dict[str, bytes]
) -> None:
    from pathlib import Path

    csrf = await login(client)
    response = await client.post(
        "/api/uploads",
        files={"file": ("evil.zip", _zip(members), "application/zip")},
        headers=csrf_headers(csrf),
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upload_rejected"
    for root in (Path(state.config.storage.data_dir) / "uploads").iterdir():
        assert not (root / "extracted").exists()
