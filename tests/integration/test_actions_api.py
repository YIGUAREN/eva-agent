import hashlib
import json
from pathlib import Path

from httpx import AsyncClient

from eva_agent.domain.run import RunStatus
from eva_agent.server.state import AppState
from tests.support.api import PASTE_PAYLOAD, csrf_headers, login


async def _create(client: AsyncClient, state: AppState) -> str:
    csrf = await login(client)
    response = await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))
    return response.json()["run_id"]


async def _finish_pending(state: AppState, run_id: str) -> None:
    """Simulate the worker draining the start command without running the graph."""
    for command in await state.commands.list_for_run(run_id):
        if command.status == "pending":
            await state.commands.finish(command.id, "done")


def _keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [key for key, item in value.items() for key in (key, *_keys(item))]
    if isinstance(value, list):
        return [key for item in value for key in _keys(item)]
    return []


async def test_approve_requires_waiting_approval(client: AsyncClient, state: AppState) -> None:
    run_id = await _create(client, state)
    csrf = await login(client)

    response = await client.post(
        f"/api/runs/{run_id}/approve",
        json={"approved": True},
        headers=csrf_headers(csrf),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_state"
    events = await state.audit.list_for_run(run_id)
    assert [(event.action, event.outcome) for event in events] == [("run.approve", "rejected")]


async def test_approve_enqueues_a_command_and_audits(client: AsyncClient, state: AppState) -> None:
    run_id = await _create(client, state)
    await _finish_pending(state, run_id)
    await state.runs.set_status(run_id, RunStatus.WAITING_APPROVAL)
    csrf = await login(client)

    response = await client.post(
        f"/api/runs/{run_id}/approve",
        json={"approved": True, "plan": {"objective": "edited"}},
        headers=csrf_headers(csrf),
    )

    assert response.status_code == 202
    commands = await state.commands.list_for_run(run_id)
    assert commands[-1].kind == "approve"
    assert commands[-1].payload_json == {"approved": True, "plan": {"objective": "edited"}}
    assert [event.action for event in await state.audit.list_for_run(run_id)] == ["run.approve"]


async def test_approve_with_a_pending_command_is_audited_as_rejected(client: AsyncClient, state: AppState) -> None:
    run_id = await _create(client, state)
    # The start command enqueued at creation is still pending.
    await state.runs.set_status(run_id, RunStatus.WAITING_APPROVAL)
    csrf = await login(client)

    response = await client.post(
        f"/api/runs/{run_id}/approve",
        json={"approved": True},
        headers=csrf_headers(csrf),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "command_pending"
    events = await state.audit.list_for_run(run_id)
    assert [(event.action, event.outcome) for event in events] == [("run.approve", "rejected")]


async def test_cancel_enqueues_a_command(client: AsyncClient, state: AppState) -> None:
    run_id = await _create(client, state)
    csrf = await login(client)

    response = await client.post(f"/api/runs/{run_id}/cancel", headers=csrf_headers(csrf))

    assert response.status_code == 202
    commands = await state.commands.list_for_run(run_id)
    assert commands[-1].kind == "cancel"


async def test_cancel_of_a_completed_run_is_rejected(client: AsyncClient, state: AppState) -> None:
    run_id = await _create(client, state)
    await state.runs.set_status(run_id, RunStatus.COMPLETED)
    csrf = await login(client)

    response = await client.post(f"/api/runs/{run_id}/cancel", headers=csrf_headers(csrf))

    assert response.status_code == 409
    events = await state.audit.list_for_run(run_id)
    assert [(event.action, event.outcome) for event in events] == [("run.cancel", "rejected")]


async def test_report_is_404_until_the_worker_snapshots_it(client: AsyncClient, state: AppState) -> None:
    run_id = await _create(client, state)
    csrf = await login(client)

    missing = await client.get(f"/api/runs/{run_id}/report")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "report_not_ready"

    await state.reports.upsert(run_id, 4, {"run_id": run_id, "version": 4}, "# report")
    ready = await client.get(f"/api/runs/{run_id}/report", headers=csrf_headers(csrf))
    assert ready.status_code == 200
    assert ready.json()["version"] == 4
    assert ready.json()["markdown"] == "# report"


async def test_public_report_requires_publishing(client: AsyncClient, state: AppState) -> None:
    run_id = await _create(client, state)
    await state.reports.upsert(run_id, 4, {"run_id": run_id}, "# report")

    assert (await client.get(f"/api/public/runs/{run_id}/report")).status_code == 404

    await state.runs.set_status(run_id, RunStatus.COMPLETED)
    await state.runs.set_public(run_id, True, now=state.now())
    assert (await client.get(f"/api/public/runs/{run_id}/report")).status_code == 200


async def test_public_report_sanitizes_budget_and_paths(client: AsyncClient, state: AppState) -> None:
    run_id = await _create(client, state)
    payload = {
        "run_id": run_id,
        "version": 5,
        "budget": {"max_cost_usd": 3.0},
        "nested": {"raw_result_path": "/tmp/raw.json", "keep": "value"},
        "probe": {"path": "C:/probe/target.jsonl", "total": 3},
        "dimensions": [
            {"name": "correctness", "evidence": [{"locator": "D:\\data\\ds.jsonl#3"}]},
        ],
        "findings": [
            {"title": "absolute path leak", "evidence": [{"locator": "/data/ds.jsonl#3"}]},
        ],
        "unc": {
            "windows": "\\\\server\\share\\f.jsonl#3",
            "posix": "//server/share/f.jsonl#3",
            "spaced": "C:\\Users\\John Doe\\ds.jsonl",
        },
        "references": [{"url": "https://gitlab.com/g/a/-/jobs/1"}],
    }
    markdown = (
        f"# report {run_id}\n"
        "- 预算：cost ≤ 3.0 美元\n"
        "- 记录文件：`C:/secret/report.md`\n"
        "- 详情：`/tmp/secret/report.md`\n"
        "- 定位：`/data/ds.jsonl#3`\n"
        "- 通过率：2/3\n"
        "- 生成时间：2026-01-01T00:00:00+00:00\n"
        "- 参考：https://example.com/eval\n"
        "- 另一参考：http://example.com/x\n"
    )
    await state.reports.upsert(run_id, 5, payload, markdown)
    await state.runs.set_status(run_id, RunStatus.COMPLETED)
    await state.runs.set_public(run_id, True, now=state.now())

    response = await client.get(f"/api/public/runs/{run_id}/report")

    assert response.status_code == 200
    body = response.json()
    report = body["report"]
    assert "budget" not in report
    assert all(not key.endswith("_path") for key in _keys(report))
    assert report["nested"]["keep"] == "value"
    assert report["dimensions"][0]["evidence"][0]["locator"] == "<path>"
    assert report["findings"][0]["evidence"][0]["locator"] == "<path>"
    assert report["unc"]["windows"] == "<path>"
    assert report["unc"]["posix"] == "<path>"
    assert report["unc"]["spaced"] == "<path>"
    assert report["references"][0]["url"] == "https://gitlab.com/g/a/-/jobs/1"
    assert "D:\\data" not in response.text
    assert "D:\\\\data" not in response.text
    assert "/data/ds.jsonl" not in response.text
    assert "server" not in response.text
    assert "share" not in response.text
    assert "John Doe" not in response.text
    assert "ds.jsonl" not in response.text
    assert "预算" not in body["markdown"]
    assert "C:/secret" not in body["markdown"]
    assert "/tmp/secret" not in body["markdown"]
    assert "/data/ds.jsonl" not in body["markdown"]
    assert "<path>#3" in body["markdown"]
    assert "2/3" in body["markdown"]
    assert "2026-01-01T00:00:00+00:00" in body["markdown"]
    assert "https://example.com/eval" in body["markdown"]
    assert "http://example.com/x" in body["markdown"]
    expected = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    assert body["sha256"] == expected

    # Admin reads are unsanitized and must keep the raw locator values.
    csrf = await login(client)
    admin = await client.get(f"/api/runs/{run_id}/report", headers=csrf_headers(csrf))
    assert admin.status_code == 200
    admin_report = admin.json()["report"]
    assert admin_report["dimensions"][0]["evidence"][0]["locator"] == "D:\\data\\ds.jsonl#3"
    assert admin_report["findings"][0]["evidence"][0]["locator"] == "/data/ds.jsonl#3"
    assert admin_report["unc"]["windows"] == "\\\\server\\share\\f.jsonl#3"
    assert admin_report["unc"]["posix"] == "//server/share/f.jsonl#3"
    assert admin_report["unc"]["spaced"] == "C:\\Users\\John Doe\\ds.jsonl"


async def test_artifact_download_respects_visibility(client: AsyncClient, state: AppState, tmp_path: Path) -> None:
    run_id = await _create(client, state)
    await state.runs.set_status(run_id, RunStatus.COMPLETED)
    await state.runs.set_public(run_id, True, now=state.now())
    run_root = Path(state.config.storage.data_dir) / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "report.md").write_text("# public", encoding="utf-8")
    (run_root / "plan.approved.yaml").write_text("objective: secret", encoding="utf-8")
    await state.artifacts.add(
        artifact_id="art_public",
        run_id=run_id,
        name="report.md",
        sha256="0" * 64,
        size_bytes=8,
        content_type="text/markdown",
        visibility="admin",
    )
    await state.artifacts.add(
        artifact_id="art_admin",
        run_id=run_id,
        name="plan.approved.yaml",
        sha256="0" * 64,
        size_bytes=18,
        content_type="application/yaml",
        visibility="admin",
    )
    csrf = await login(client)

    assert (await client.get(f"/api/public/runs/{run_id}/artifacts/report.md")).status_code == 404
    assert (await client.get(f"/api/public/runs/{run_id}/artifacts/plan.approved.yaml")).status_code == 404
    admin = await client.get(f"/api/runs/{run_id}/artifacts/report.md", headers=csrf_headers(csrf))
    assert admin.status_code == 200
    admin_plan = await client.get(f"/api/runs/{run_id}/artifacts/plan.approved.yaml", headers=csrf_headers(csrf))
    assert admin_plan.status_code == 200
    assert "secret" in admin_plan.text


async def test_graders_and_demo_targets_are_public(client: AsyncClient) -> None:
    graders = await client.get("/api/graders")
    targets = await client.get("/api/demo-targets")

    assert graders.status_code == 200
    assert len(graders.json()["graders"]) > 0
    assert targets.status_code == 200
    assert all("api_key_env" not in item for item in targets.json()["targets"])
