import httpx
import pytest

from eva_agent.adapters.aig.attachments import attachment_names
from eva_agent.adapters.aig.client import AigClient
from eva_agent.adapters.aig.errors import AigUnavailable
from eva_agent.config.settings import AigSettings

CSV = b"\xef\xbb\xbfstatus,modelName\ncompleted,demo\n"


def payload(**content: object) -> dict:
    return {"result": {"msgType": "json", "content": [content]}}


def test_attachment_names_reads_the_observed_field() -> None:
    item = {"attachment": "attachment_0123456789abcdef0123456789abcdef_1793422813849140825_1793422813.csv"}
    assert attachment_names(payload(**item)) == [item["attachment"]]


def test_attachment_names_ignores_missing_and_unsafe_names() -> None:
    assert attachment_names({"result": {"content": [{}]}}) == []
    assert attachment_names({"result": {"content": [{"attachment": "../evil.csv"}]}}) == []
    assert attachment_names({"result": {"content": [{"attachment": "notes.txt"}]}}) == []
    assert attachment_names({"result": "nope"}) == []
    assert attachment_names({}) == []


def client(handler) -> AigClient:
    return AigClient(
        AigSettings(base_url="http://127.0.0.1:8088"),
        http=httpx.AsyncClient(base_url="http://127.0.0.1:8088", transport=httpx.MockTransport(handler)),
    )


async def test_download_posts_the_per_session_path() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=CSV)

    name = "attachment_0123456789abcdef0123456789abcdef_1_2.csv"
    content = await client(handler).download("session-1", name)

    assert content == CSV
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/v1/app/tasks/session-1/downloadFile"
    assert seen[0].read().decode() == f'{{"fileUrl":"{name}"}}'


@pytest.mark.parametrize(
    "session_id,name",
    [
        ("../etc", "attachment_1_2_3.csv"),
        ("session-1", "../etc.csv"),
        ("session-1", "notes.txt"),
        ("session-1", "a" * 300 + ".csv"),
    ],
)
async def test_download_rejects_unsafe_names(session_id: str, name: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("unsafe names must not reach the network")

    with pytest.raises(AigUnavailable):
        await client(handler).download(session_id, name)


async def test_download_rejects_an_html_error_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>error</html>")

    with pytest.raises(AigUnavailable):
        await client(handler).download("session-1", "attachment_1_2_3.csv")
