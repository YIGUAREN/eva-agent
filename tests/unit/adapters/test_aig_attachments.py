from collections.abc import AsyncIterator

import httpx
import pytest

from eva_agent.adapters.aig.attachments import MAX_ATTACHMENT_BYTES, attachment_names
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


def test_attachment_names_rejects_a_leading_dot() -> None:
    assert attachment_names(payload(attachment="..csv")) == []
    assert attachment_names(payload(attachment=".hidden.csv")) == []
    assert attachment_names(payload(attachment="results.v2.csv")) == ["results.v2.csv"]


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


async def test_download_rejects_a_non_success_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal error")

    with pytest.raises(AigUnavailable):
        await client(handler).download("session-1", "attachment_1_2_3.csv")


async def test_download_rejects_an_empty_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"", headers={"content-type": "text/csv"})

    with pytest.raises(AigUnavailable):
        await client(handler).download("session-1", "attachment_1_2_3.csv")


class CountingStream(httpx.AsyncByteStream):
    """Yield pre-built chunks one at a time and count how many were pulled."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.pulled = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            self.pulled += 1
            yield chunk


async def test_download_stops_an_oversized_body_mid_stream() -> None:
    # A distinctive marker proves the raised message never carries body content.
    marker = b"SECRET-CHUNK"
    per_chunk = 1024 * 1024
    chunk = marker + b"x" * (per_chunk - len(marker))
    chunk_count = 2 * (MAX_ATTACHMENT_BYTES // per_chunk)  # twice the cap, in many chunks
    stream = CountingStream([chunk] * chunk_count)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/csv"})

    with pytest.raises(AigUnavailable) as excinfo:
        await client(handler).download("session-1", "attachment_1_2_3.csv")

    # The guard fires during the read, not after every chunk was buffered.
    assert 0 < stream.pulled < chunk_count
    message = str(excinfo.value)
    assert marker.decode() not in message
    assert "200" not in message
    assert "OK" not in message


@pytest.mark.parametrize("media_type", ["text/html", "application/json"])
async def test_download_rejects_document_content_types(media_type: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=CSV, headers={"content-type": media_type})

    with pytest.raises(AigUnavailable):
        await client(handler).download("session-1", "attachment_1_2_3.csv")


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"content-type": "text/csv"},
        {"content-type": "text/csv; charset=utf-8"},
        {"content-type": "application/octet-stream"},
        # A BOM-prefixed CSV served through Go's http.ServeContent commonly comes
        # back as text/plain; rejecting it would reject every real attachment.
        {"content-type": "text/plain; charset=utf-8"},
    ],
)
async def test_download_accepts_csv_like_content_types(headers: dict[str, str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=CSV, headers=headers)

    assert await client(handler).download("session-1", "attachment_1_2_3.csv") == CSV
