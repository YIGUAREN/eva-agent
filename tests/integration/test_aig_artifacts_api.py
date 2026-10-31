from pathlib import Path

from httpx import AsyncClient

from eva_agent.domain.run import RunStatus
from eva_agent.server.state import AppState
from tests.support.api import csrf_headers, login

PASTE_PAYLOAD = {
    "input": {"kind": "paste", "paste": {"query": "what is 2+2?", "response": "4"}, "objective": "评估回答"},
    "budget": {"max_cost_usd": 1.0, "max_wall_clock_s": 60.0, "max_target_requests": 1},
}


async def _aig_run(client: AsyncClient, state: AppState) -> str:
    csrf = await login(client)
    created = await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))
    assert created.status_code == 201, created.text
    run_id = created.json()["run_id"]
    await state.runs.set_status(run_id, RunStatus.COMPLETED)
    await state.runs.set_public(run_id, True, now=state.now())
    aig_dir = Path(state.config.storage.data_dir) / "runs" / run_id / "aig"
    aig_dir.mkdir(parents=True, exist_ok=True)
    (aig_dir / "sess-abc.json").write_text('{"session_id": "sess-abc"}', encoding="utf-8")
    (aig_dir / "sess-abc.csv").write_text("id,score\n1,0\n", encoding="utf-8")
    for name, content_type in (
        ("aig/sess-abc.json", "application/json"),
        ("aig/sess-abc.csv", "text/csv; charset=utf-8"),
    ):
        await state.artifacts.add(
            artifact_id=f"art_{name.replace('/', '_').replace('.', '_')}",
            run_id=run_id,
            name=name,
            sha256="0" * 64,
            size_bytes=10,
            content_type=content_type,
            visibility="admin",
        )
    return run_id


async def test_admin_downloads_aig_raw_result_and_csv(client: AsyncClient, state: AppState) -> None:
    run_id = await _aig_run(client, state)
    csrf = await login(client)

    raw = await client.get(f"/api/runs/{run_id}/artifacts/aig/sess-abc.json", headers=csrf_headers(csrf))
    assert raw.status_code == 200, raw.text
    assert "sess-abc" in raw.text

    exported = await client.get(f"/api/runs/{run_id}/artifacts/aig/sess-abc.csv", headers=csrf_headers(csrf))
    assert exported.status_code == 200, exported.text
    assert exported.headers["content-type"].startswith("text/csv")
    assert "id,score" in exported.text


async def test_public_cannot_reach_aig_artifacts(client: AsyncClient, state: AppState) -> None:
    run_id = await _aig_run(client, state)

    assert (await client.get(f"/api/public/runs/{run_id}/artifacts/aig/sess-abc.json")).status_code == 404
    assert (await client.get(f"/api/public/runs/{run_id}/artifacts/aig/sess-abc.csv")).status_code == 404


async def test_unknown_aig_session_is_404(client: AsyncClient, state: AppState) -> None:
    run_id = await _aig_run(client, state)
    csrf = await login(client)

    missing = await client.get(f"/api/runs/{run_id}/artifacts/aig/sess-zzz.json", headers=csrf_headers(csrf))
    assert missing.status_code == 404


async def test_traversal_shaped_name_is_rejected(client: AsyncClient, state: AppState) -> None:
    run_id = await _aig_run(client, state)
    csrf = await login(client)

    escaped = await client.get(f"/api/runs/{run_id}/artifacts/aig/%2E%2E/report.json", headers=csrf_headers(csrf))
    assert escaped.status_code == 404


async def test_unescaped_traversal_normalizes_without_escaping(
    client: AsyncClient, state: AppState
) -> None:
    run_id = await _aig_run(client, state)
    csrf = await login(client)

    # httpx collapses this to /artifacts/report.json, which is not registered.
    response = await client.get(f"/api/runs/{run_id}/artifacts/aig/../report.json", headers=csrf_headers(csrf))
    assert response.status_code == 404
