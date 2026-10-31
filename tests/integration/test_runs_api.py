from datetime import UTC, datetime, timedelta

from httpx import AsyncClient

from eva_agent.domain.run import RunStatus
from eva_agent.server.state import AppState
from eva_agent.storage.uploads import UPLOAD_EXTRACTING
from tests.support.api import PASTE_PAYLOAD, csrf_headers, login


async def test_create_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/runs", json=PASTE_PAYLOAD)

    assert response.status_code == 401


async def test_create_requires_csrf(client: AsyncClient) -> None:
    await login(client)

    response = await client.post("/api/runs", json=PASTE_PAYLOAD)

    assert response.status_code == 403


async def test_create_persists_a_queued_run_and_a_start_command(client: AsyncClient, state: AppState) -> None:
    csrf = await login(client)

    response = await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))

    assert response.status_code == 201, response.text
    run_id = response.json()["run_id"]
    assert response.json()["status"] == "queued"
    assert len(run_id) == 36
    row = await state.runs.get(run_id)
    assert row is not None and row.status == "queued" and row.created_by == "admin"
    commands = await state.commands.list_for_run(run_id)
    assert [(command.kind, command.status) for command in commands] == [("start", "pending")]


async def test_idempotency_key_returns_the_original_run(client: AsyncClient, state: AppState) -> None:
    csrf = await login(client)
    payload = {**PASTE_PAYLOAD, "idempotency_key": "0123456789abcdef"}

    first = await client.post("/api/runs", json=payload, headers=csrf_headers(csrf))
    second = await client.post("/api/runs", json=payload, headers=csrf_headers(csrf))

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["run_id"] == second.json()["run_id"]
    assert len(await state.runs.list_recent(10)) == 1
    assert len(await state.commands.list_for_run(first.json()["run_id"])) == 1


async def test_unknown_target_is_rejected(client: AsyncClient) -> None:
    csrf = await login(client)
    payload = {
        "input": {"kind": "target", "target_ref": "missing-target", "target_prompts": ["hi"]},
        "budget": PASTE_PAYLOAD["budget"],
    }

    response = await client.post("/api/runs", json=payload, headers=csrf_headers(csrf))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_target"


async def test_admin_list_and_detail(client: AsyncClient, state: AppState) -> None:
    csrf = await login(client)
    run_id = (await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))).json()["run_id"]
    await state.runs.set_status(run_id, RunStatus.WAITING_APPROVAL)

    listing = await client.get("/api/runs")
    detail = await client.get(f"/api/runs/{run_id}")

    assert listing.status_code == 200
    assert [item["run_id"] for item in listing.json()["items"]] == [run_id]
    assert detail.status_code == 200
    assert detail.json()["status"] == "waiting_approval"
    assert detail.json()["input"]["paste"]["query"] == "what is 2+2"


async def test_detail_of_a_missing_run_is_404(client: AsyncClient) -> None:
    await login(client)

    response = await client.get("/api/runs/run_missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_publish_requires_a_completed_run(client: AsyncClient) -> None:
    csrf = await login(client)
    run_id = (await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))).json()["run_id"]

    response = await client.post(
        f"/api/runs/{run_id}/publish",
        json={"published": True},
        headers=csrf_headers(csrf),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_completed"


async def test_publish_unpublish_and_public_visibility(client: AsyncClient, state: AppState) -> None:
    csrf = await login(client)
    run_id = (await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))).json()["run_id"]
    await state.runs.set_status(run_id, RunStatus.COMPLETED)

    hidden = await client.get(f"/api/public/runs/{run_id}")
    assert hidden.status_code == 404

    published = await client.post(
        f"/api/runs/{run_id}/publish",
        json={"published": True},
        headers=csrf_headers(csrf),
    )
    assert published.status_code == 200
    assert published.json()["is_public"] is True

    public_list = await client.get("/api/public/runs")
    assert [item["run_id"] for item in public_list.json()["items"]] == [run_id]

    visible = await client.get(f"/api/public/runs/{run_id}")
    assert visible.status_code == 200
    assert "created_by" not in visible.json()

    unpublished = await client.post(
        f"/api/runs/{run_id}/publish",
        json={"published": False},
        headers=csrf_headers(csrf),
    )
    assert unpublished.status_code == 200
    assert (await client.get(f"/api/public/runs/{run_id}")).status_code == 404


async def test_publish_is_audited(client: AsyncClient, state: AppState) -> None:
    csrf = await login(client)
    run_id = (await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))).json()["run_id"]
    await state.runs.set_status(run_id, RunStatus.COMPLETED)

    await client.post(f"/api/runs/{run_id}/publish", json={"published": True}, headers=csrf_headers(csrf))

    actions = [event.action for event in await state.audit.list_for_run(run_id)]
    assert actions == ["run.publish"]


def _upload_payload(upload_id: str) -> dict:
    return {
        "input": {"kind": "upload", "upload_id": upload_id},
        "budget": {"max_cost_usd": 1.0, "max_wall_clock_s": 60.0, "max_target_requests": 1},
    }


async def test_create_run_rejects_unknown_upload(client: AsyncClient, state: AppState) -> None:
    csrf = await login(client)
    response = await client.post(
        "/api/runs", json=_upload_payload("upload_missing"), headers=csrf_headers(csrf)
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unknown_upload"


async def test_create_run_rejects_unready_upload(client: AsyncClient, state: AppState) -> None:
    await state.uploads.create(
        "upload_pending",
        filename="p.zip",
        size_bytes=1,
        sha256="a" * 64,
        format="zip",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    assert (await state.uploads.get("upload_pending")).status == UPLOAD_EXTRACTING
    csrf = await login(client)
    response = await client.post(
        "/api/runs", json=_upload_payload("upload_pending"), headers=csrf_headers(csrf)
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "upload_not_ready"
