import json

import httpx
import pytest

from eva_agent.adapters.aig.client import AigClient
from eva_agent.adapters.aig.errors import AigRejected, AigTimeout, AigUnavailable
from eva_agent.config.settings import AigSettings

RAW = {
    "result": {
        "msgType": "json",
        "content": [{"modelName": "m", "total": 1, "jailbreak": 0, "score": 100.0}],
    }
}


def make_client(handler, **overrides) -> AigClient:
    settings = AigSettings(
        base_url="http://aig.test",
        poll_initial_s=0.001,
        poll_max_s=0.002,
        max_wait_s=0.05,
        min_wait_s=0.001,
        **overrides,
    )
    return AigClient(
        settings,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://aig.test"),
    )


async def test_create_wait_and_result_contract() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/api/v1/app/taskapi/tasks":
            body = json.loads(request.content)
            assert body["type"] == "model_redteam_report"
            return httpx.Response(200, json={"status": 0, "message": "ok", "data": {"session_id": "s1"}})
        if request.url.path == "/api/v1/app/taskapi/status/s1":
            return httpx.Response(200, json={"status": 0, "message": "ok", "data": {"status": "completed"}})
        if request.url.path == "/api/v1/app/taskapi/result/s1":
            return httpx.Response(200, json={"status": 0, "message": "ok", "data": RAW})
        return httpx.Response(404)

    client = make_client(handler)
    session = await client.create({"type": "model_redteam_report", "content": {}})
    assert session == "s1"
    assert await client.wait(session, max_wait_s=0.05) == "completed"
    data = await client.result(session)
    assert data["result"]["msgType"] == "json"
    assert seen == [
        ("POST", "/api/v1/app/taskapi/tasks"),
        ("GET", "/api/v1/app/taskapi/status/s1"),
        ("GET", "/api/v1/app/taskapi/result/s1"),
    ]


async def test_wait_polls_until_completed() -> None:
    states = iter(["pending", "running", "completed"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "data": {"status": next(states)}})

    assert await make_client(handler).wait("s1", max_wait_s=0.05) == "completed"


async def test_failed_task_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "data": {"status": "failed"}})

    with pytest.raises(AigRejected, match="failed"):
        await make_client(handler).wait("s1", max_wait_s=0.05)


@pytest.mark.parametrize("status", ["pending", "running", "todo", "doing"])
async def test_wait_polls_through_in_progress_statuses(status: str) -> None:
    states = iter([status, "done"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "data": {"status": next(states)}})

    assert await make_client(handler).wait("s1", max_wait_s=0.05) == "done"


@pytest.mark.parametrize("status", ["completed", "done"])
async def test_success_status_returns(status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "data": {"status": status}})

    assert await make_client(handler).wait("s1", max_wait_s=0.05) == status


@pytest.mark.parametrize("status", ["failed", "error", "terminated"])
async def test_failure_status_is_rejected_without_message_echo(status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "data": {"status": status, "message": "secret sk-123"}})

    with pytest.raises(AigRejected, match="failed") as exc:
        await make_client(handler).wait("s1", max_wait_s=0.05)
    assert "sk-123" not in str(exc.value)


async def test_unknown_status_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "data": {"status": "weird"}})

    with pytest.raises(AigRejected, match="unknown scan status"):
        await make_client(handler).wait("s1", max_wait_s=0.05)


async def test_wait_times_out() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "data": {"status": "running"}})

    with pytest.raises(AigTimeout):
        await make_client(handler).wait("s1", max_wait_s=0.02)


async def test_http_error_is_unavailable_without_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="upstream secret sk-123")

    with pytest.raises(AigUnavailable) as exc:
        await make_client(handler).result("s1")
    assert "sk-123" not in str(exc.value)


async def test_envelope_failure_is_rejected_without_message_echo() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 1, "message": "bad token sk-123", "data": None})

    with pytest.raises(AigRejected) as exc:
        await make_client(handler).create({"type": "model_redteam_report", "content": {}})
    assert "sk-123" not in str(exc.value)


async def test_transport_error_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(AigUnavailable):
        await make_client(handler).create({"type": "model_redteam_report", "content": {}})


async def test_invalid_session_id_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 0, "data": {"session_id": "../escape"}})

    with pytest.raises(AigUnavailable) as exc:
        await make_client(handler).create({"type": "model_redteam_report", "content": {}})
    assert "../escape" not in str(exc.value)
