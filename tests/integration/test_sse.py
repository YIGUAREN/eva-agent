import json
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from eva_agent.config.settings import AigSettings
from eva_agent.domain.run import RunStatus
from eva_agent.server.state import AppState
from tests.support.api import PASTE_PAYLOAD, csrf_headers, login
from tests.support.runtime import fake_config
from tests.support.server import build_test_app


async def _completed_run(client: AsyncClient, state: AppState) -> str:
    csrf = await login(client)
    run_id = (await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))).json()["run_id"]
    await state.runs.set_status(run_id, RunStatus.COMPLETED)
    await state.runs.set_public(run_id, True, now=state.now())
    # Seq 1 is the run.created event written by POST /api/runs.
    await state.runs.append_event(run_id, "report.ready", {"markdown_path": "C:/secret/report.md"})
    return run_id


def _events(body: str) -> list[dict[str, object]]:
    parsed = []
    for block in body.strip().split("\n\n"):
        data = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
        if data:
            parsed.append(json.loads(data[-1]))
    return parsed


async def test_sse_replays_all_events_for_a_completed_run(
    client: AsyncClient, state: AppState
) -> None:
    run_id = await _completed_run(client, state)
    csrf = await login(client)

    response = await client.get(f"/api/runs/{run_id}/events", headers=csrf_headers(csrf))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert [event["seq"] for event in events] == [1, 2]
    assert events[0]["type"] == "run.created"


async def test_sse_resumes_after_last_event_id(client: AsyncClient, state: AppState) -> None:
    run_id = await _completed_run(client, state)
    csrf = await login(client)

    response = await client.get(
        f"/api/runs/{run_id}/events",
        headers={**csrf_headers(csrf), "Last-Event-ID": "1"},
    )

    assert [event["seq"] for event in _events(response.text)] == [2]


async def test_public_sse_strips_paths_and_requires_publishing(
    client: AsyncClient, state: AppState
) -> None:
    run_id = await _completed_run(client, state)

    response = await client.get(f"/api/public/runs/{run_id}/events")

    assert response.status_code == 200
    assert "C:/secret" not in response.text
    assert _events(response.text)[1]["payload"] == {}

    await state.runs.set_public(run_id, False, now=state.now())
    assert (await client.get(f"/api/public/runs/{run_id}/events")).status_code == 404


async def test_public_sse_redacts_the_configured_internal_host(tmp_path: Path) -> None:
    config = fake_config(tmp_path).model_copy(
        update={"aig": AigSettings(base_url="http://aig.internal:8100")}
    )
    app, state = await build_test_app(tmp_path, config=config)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as client:
            run_id = await _completed_run(client, state)
            await state.runs.append_event(
                run_id,
                "command.failed",
                {"error": "cannot reach http://aig.internal:8100/v1 with sk-secret12345"},
            )

            response = await client.get(f"/api/public/runs/{run_id}/events")

        assert response.status_code == 200
        assert "aig.internal" not in response.text
        assert "sk-secret12345" not in response.text
        assert "<internal>" in response.text
        assert "<secret>" in response.text
    finally:
        await state.engine.dispose()


async def test_sse_requires_authentication_for_admin_routes(client: AsyncClient, state: AppState) -> None:
    run_id = await _completed_run(client, state)
    client.cookies.clear()

    response = await client.get(f"/api/runs/{run_id}/events")

    assert response.status_code == 401
