import pytest
from httpx import AsyncClient

from eva_agent.domain.run import RunStatus
from eva_agent.server.routes.reports import public_report_usage
from eva_agent.server.schemas import sanitize_public_payload
from eva_agent.server.state import AppState
from tests.support.api import PASTE_PAYLOAD, csrf_headers, login


async def _create(client: AsyncClient) -> str:
    csrf = await login(client)
    response = await client.post("/api/runs", json=PASTE_PAYLOAD, headers=csrf_headers(csrf))
    assert response.status_code == 201, response.text
    return response.json()["run_id"]


async def _publish(state: AppState, run_id: str) -> None:
    await state.runs.set_status(run_id, RunStatus.COMPLETED)
    await state.runs.set_public(run_id, True, now=state.now())


def test_public_payload_keeps_only_the_four_usage_keys() -> None:
    payload = {
        "usage": {
            "estimated": True,
            "cached": 2,
            "fresh": 1,
            "budget_enforced": True,
            "spent_usd": 0.9,
        },
        "active_scan": {"kind": "skill_scan", "raw_result_path": "/srv/aig/x.json"},
    }
    assert public_report_usage(payload) == {
        "estimated": True,
        "cached": 2,
        "fresh": 1,
        "budget_enforced": True,
    }
    public = sanitize_public_payload(payload)
    assert "raw_result_path" not in public["active_scan"]


async def test_public_report_strips_spend_and_keeps_admin_usage(
    client: AsyncClient, state: AppState
) -> None:
    run_id = await _create(client)
    stored_usage = {
        "estimated": True,
        "cached": 2,
        "fresh": 1,
        "budget_enforced": True,
        "spent_usd": 12.345678,
        "debug_internal": "admin-only",
    }
    payload = {
        "run_id": run_id,
        "version": 4,
        "usage": stored_usage,
        "truncated_dimensions": ["quality"],
        "truncated_sample_counts": [
            {
                "dimension": "quality",
                "sample_count": 5,
                "completed_samples": 2,
                "missing_samples": 3,
            }
        ],
    }
    markdown = (
        f"# report {run_id}\n"
        "- 预算：cost ≤ 3.0 美元\n"
        "- 已花费：12.345678 USD\n"
        "- 截断维度：quality\n"
        "- 评分来源：缓存 2，新评分 1\n"
        "| quality | 5 | 2 | 3 |\n"
    )
    await state.reports.upsert(run_id, 4, payload, markdown)
    await _publish(state, run_id)

    response = await client.get(f"/api/public/runs/{run_id}/report")

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["usage"] == {
        "estimated": True,
        "cached": 2,
        "fresh": 1,
        "budget_enforced": True,
    }
    assert "spent_usd" not in response.text
    assert "debug_internal" not in response.text
    assert "12.345678" not in response.text
    assert "- 已花费：" not in body["markdown"]
    assert "- 预算：" not in body["markdown"]
    # Sample counts are not spend and carry no paths or hosts, so they remain
    # public; the Markdown truncation table stays readable.
    assert body["report"]["truncated_sample_counts"] == payload["truncated_sample_counts"]
    assert "| quality | 5 | 2 | 3 |" in body["markdown"]

    csrf = await login(client)
    admin = await client.get(f"/api/runs/{run_id}/report", headers=csrf_headers(csrf))

    assert admin.status_code == 200
    assert admin.json()["report"]["usage"] == stored_usage
    assert "- 已花费：12.345678 USD" in admin.json()["markdown"]


@pytest.mark.parametrize(
    "usage",
    [
        ["spent_usd", 12.345678],
        "spent_usd=12.345678",
    ],
)
async def test_public_report_neutralizes_non_mapping_usage(
    client: AsyncClient, state: AppState, usage: object
) -> None:
    run_id = await _create(client)
    await state.reports.upsert(
        run_id, 4, {"run_id": run_id, "version": 4, "usage": usage}, "# report"
    )
    await _publish(state, run_id)

    response = await client.get(f"/api/public/runs/{run_id}/report")

    assert response.status_code == 200
    assert response.json()["report"]["usage"] == {}
    assert "spent_usd" not in response.text
    assert "12.345678" not in response.text


async def test_public_report_whitelists_truncated_sample_count_entries(
    client: AsyncClient, state: AppState
) -> None:
    run_id = await _create(client)
    payload = {
        "run_id": run_id,
        "version": 4,
        "truncated_dimensions": ["quality"],
        "truncated_sample_counts": [
            {
                "dimension": "quality",
                "sample_count": 5,
                "completed_samples": 2,
                "missing_samples": 3,
                "spent_usd": 12.345678,
                "raw_result_path": "/srv/aig/x.json",
            }
        ],
    }
    await state.reports.upsert(run_id, 4, payload, "# report")
    await _publish(state, run_id)

    response = await client.get(f"/api/public/runs/{run_id}/report")

    assert response.status_code == 200
    assert response.json()["report"]["truncated_sample_counts"] == [
        {
            "dimension": "quality",
            "sample_count": 5,
            "completed_samples": 2,
            "missing_samples": 3,
        }
    ]
    assert "spent_usd" not in response.text
    assert "12.345678" not in response.text


async def test_public_report_does_not_inject_usage_into_legacy_payloads(
    client: AsyncClient, state: AppState
) -> None:
    run_id = await _create(client)
    await state.reports.upsert(run_id, 3, {"run_id": run_id, "version": 3}, "# report")
    await _publish(state, run_id)

    response = await client.get(f"/api/public/runs/{run_id}/report")

    assert response.status_code == 200
    assert "usage" not in response.json()["report"]


async def test_public_report_drops_hostile_usage_values_but_admin_keeps_them(
    client: AsyncClient, state: AppState
) -> None:
    run_id = await _create(client)
    hostile_usage = {
        "estimated": ["spent_usd", 12.345678],
        "cached": {"spent_usd": 12.345678},
        "fresh": "spent_usd=12.345678",
        "budget_enforced": "spent_usd=12.345678",
    }
    await state.reports.upsert(
        run_id, 4, {"run_id": run_id, "version": 4, "usage": hostile_usage}, "# report"
    )
    await _publish(state, run_id)

    response = await client.get(f"/api/public/runs/{run_id}/report")

    assert response.status_code == 200
    assert response.json()["report"]["usage"] == {}
    assert "spent_usd" not in response.text
    assert "12.345678" not in response.text

    csrf = await login(client)
    admin = await client.get(f"/api/runs/{run_id}/report", headers=csrf_headers(csrf))

    assert admin.status_code == 200
    assert admin.json()["report"]["usage"] == hostile_usage


async def test_public_report_keeps_valid_usage_values_and_drops_invalid_ones(
    client: AsyncClient, state: AppState
) -> None:
    run_id = await _create(client)
    stored_usage = {
        "estimated": True,
        "cached": "12.345678",
        "fresh": 3,
        "budget_enforced": {"spent_usd": 12.345678},
    }
    await state.reports.upsert(
        run_id, 4, {"run_id": run_id, "version": 4, "usage": stored_usage}, "# report"
    )
    await _publish(state, run_id)

    response = await client.get(f"/api/public/runs/{run_id}/report")

    assert response.status_code == 200
    assert response.json()["report"]["usage"] == {"estimated": True, "fresh": 3}
    assert "12.345678" not in response.text


@pytest.mark.parametrize(
    "entry",
    [
        # Hostile dimension value.
        {
            "dimension": {"spent_usd": 12.345678},
            "sample_count": 5,
            "completed_samples": 2,
            "missing_samples": 3,
        },
        # Hostile count value.
        {
            "dimension": "quality",
            "sample_count": ["spent_usd", 12.345678],
            "completed_samples": 2,
            "missing_samples": 3,
        },
        # Booleans are not counts.
        {
            "dimension": "quality",
            "sample_count": True,
            "completed_samples": 2,
            "missing_samples": 3,
        },
        # Negative counts are not counts.
        {
            "dimension": "quality",
            "sample_count": 5,
            "completed_samples": -2,
            "missing_samples": 3,
        },
    ],
)
async def test_public_report_drops_hostile_truncation_entries_but_admin_keeps_them(
    client: AsyncClient, state: AppState, entry: dict[str, object]
) -> None:
    run_id = await _create(client)
    payload = {
        "run_id": run_id,
        "version": 4,
        "truncated_dimensions": ["quality"],
        "truncated_sample_counts": [entry],
    }
    await state.reports.upsert(run_id, 4, payload, "# report")
    await _publish(state, run_id)

    response = await client.get(f"/api/public/runs/{run_id}/report")

    assert response.status_code == 200
    assert response.json()["report"]["truncated_sample_counts"] == []
    assert "spent_usd" not in response.text
    assert "12.345678" not in response.text

    csrf = await login(client)
    admin = await client.get(f"/api/runs/{run_id}/report", headers=csrf_headers(csrf))

    assert admin.status_code == 200
    assert admin.json()["report"]["truncated_sample_counts"] == [entry]
