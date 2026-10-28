import ssl
from dataclasses import dataclass, field
from typing import Any

import httpcore
import pytest
from httpcore import AsyncNetworkBackend, AsyncNetworkStream

from eva_agent.adapters.target.base import TargetRequest, TargetResponseError
from eva_agent.adapters.target.http import (
    HttpCoreTransport,
    PinnedNetworkBackend,
    TargetResponseTooLargeError,
    read_body,
)
from eva_agent.security.target_policy import SecurityPolicyError, ValidatedTarget


class FakeStream(AsyncNetworkStream):
    """Minimal AsyncNetworkStream used only by PinnedNetworkBackend unit tests."""

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return b""

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        return self

    def get_extra_info(self, info: str) -> Any:
        return None


class FakeBackend(AsyncNetworkBackend):
    def __init__(self, errors: dict[str, Exception] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.errors = errors or {}

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        self.calls.append((host, port))
        if host in self.errors:
            raise self.errors[host]
        return FakeStream()

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        raise AssertionError("unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        return None


async def test_backend_connects_to_pinned_ip():
    inner = FakeBackend()
    backend = PinnedNetworkBackend(host="api.example.com", addresses=("93.184.216.34",), inner=inner)

    await backend.connect_tcp("api.example.com", 443)

    assert inner.calls == [("93.184.216.34", 443)]


async def test_backend_rejects_other_hosts():
    backend = PinnedNetworkBackend(host="api.example.com", addresses=("93.184.216.34",), inner=FakeBackend())

    with pytest.raises(SecurityPolicyError, match="unexpected host"):
        await backend.connect_tcp("evil.example.com", 443)


async def test_backend_tries_each_pinned_address():
    inner = FakeBackend(errors={"1.1.1.1": httpcore.ConnectError("connection refused")})
    backend = PinnedNetworkBackend(host="api.example.com", addresses=("1.1.1.1", "93.184.216.34"), inner=inner)

    await backend.connect_tcp("api.example.com", 443)

    assert inner.calls == [("1.1.1.1", 443), ("93.184.216.34", 443)]


async def test_backend_tries_next_address_after_connect_timeout():
    inner = FakeBackend(errors={"1.1.1.1": httpcore.ConnectTimeout("connect timed out")})
    backend = PinnedNetworkBackend(host="api.example.com", addresses=("1.1.1.1", "93.184.216.34"), inner=inner)

    await backend.connect_tcp("api.example.com", 443)

    assert inner.calls == [("1.1.1.1", 443), ("93.184.216.34", 443)]


async def test_backend_surfaces_last_connection_error():
    inner = FakeBackend(
        errors={
            "1.1.1.1": httpcore.ConnectError("first failure"),
            "93.184.216.34": httpcore.ConnectError("second failure"),
        }
    )
    backend = PinnedNetworkBackend(
        host="api.example.com",
        addresses=("1.1.1.1", "93.184.216.34"),
        inner=inner,
    )

    with pytest.raises(TargetResponseError, match="api.example.com") as excinfo:
        await backend.connect_tcp("api.example.com", 443)

    assert "second failure" in str(excinfo.value)
    assert inner.calls == [("1.1.1.1", 443), ("93.184.216.34", 443)]


async def test_backend_propagates_connect_timeout_when_all_addresses_time_out():
    inner = FakeBackend(
        errors={
            "1.1.1.1": httpcore.ConnectTimeout("first timed out"),
            "93.184.216.34": httpcore.ConnectTimeout("second timed out"),
        }
    )
    backend = PinnedNetworkBackend(
        host="api.example.com",
        addresses=("1.1.1.1", "93.184.216.34"),
        inner=inner,
    )

    with pytest.raises(httpcore.ConnectTimeout, match="second timed out"):
        await backend.connect_tcp("api.example.com", 443)

    assert inner.calls == [("1.1.1.1", 443), ("93.184.216.34", 443)]


async def test_backend_rejects_empty_address_list():
    backend = PinnedNetworkBackend(host="api.example.com", addresses=(), inner=FakeBackend())

    with pytest.raises(TargetResponseError, match="no pinned address available for api.example.com"):
        await backend.connect_tcp("api.example.com", 443)


class FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def aiter_stream(self):
        for chunk in self._chunks:
            yield chunk


async def test_read_body_enforces_limit():
    with pytest.raises(TargetResponseTooLargeError, match="response exceeds"):
        await read_body(FakeResponse([b"ab", b"cd"]), max_bytes=3)


async def test_read_body_returns_bytes():
    assert await read_body(FakeResponse([b"ab", b"cd"]), max_bytes=4) == b"abcd"


class FakeNetworkStream(AsyncNetworkStream):
    """In-memory AsyncNetworkStream: serves canned reads and records writes/TLS."""

    def __init__(self, reads: list[bytes]) -> None:
        self._reads = list(reads)
        self.written = bytearray()
        self.read_timeouts: list[float | None] = []
        self.write_timeouts: list[float | None] = []
        self.tls_server_hostnames: list[str | None] = []
        self.closed = False

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.read_timeouts.append(timeout)
        if not self._reads:
            return b""
        chunk = self._reads.pop(0)
        if len(chunk) > max_bytes:
            self._reads.insert(0, chunk[max_bytes:])
            return chunk[:max_bytes]
        return chunk

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.write_timeouts.append(timeout)
        self.written.extend(buffer)

    async def aclose(self) -> None:
        self.closed = True

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> AsyncNetworkStream:
        self.tls_server_hostnames.append(server_hostname)
        return self

    def get_extra_info(self, info: str) -> Any:
        return None


@dataclass
class ConnectCall:
    host: str
    port: int
    timeout: float | None = None


@dataclass
class RecordingBackend(AsyncNetworkBackend):
    reads: list[bytes]
    errors: dict[str, Exception] = field(default_factory=dict)
    connect_calls: list[ConnectCall] = field(default_factory=list)
    streams: list[FakeNetworkStream] = field(default_factory=list)

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        self.connect_calls.append(ConnectCall(host=host, port=port, timeout=timeout))
        if host in self.errors:
            raise self.errors[host]
        stream = FakeNetworkStream(self.reads)
        self.streams.append(stream)
        return stream

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        raise AssertionError("unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        return None


def validated_target(
    *,
    url: str = "https://api.example.com/v1/chat",
    scheme: str = "https",
    host: str = "api.example.com",
    port: int = 443,
    addresses: tuple[str, ...] = ("93.184.216.34",),
) -> ValidatedTarget:
    return ValidatedTarget(url=url, scheme=scheme, host=host, port=port, addresses=addresses, demo=False)


def http_request() -> TargetRequest:
    return TargetRequest(
        method="POST",
        url="https://api.example.com/v1/chat",
        headers={"content-type": "application/json"},
        body=b'{"x":1}',
    )


def response_bytes(
    *,
    status_line: bytes = b"HTTP/1.1 200 OK",
    headers: list[tuple[bytes, bytes]] | None = None,
    body: bytes = b"",
) -> bytes:
    head = status_line + b"\r\n"
    for name, value in headers or []:
        head += name + b": " + value + b"\r\n"
    return head + b"\r\n" + body


async def test_send_pins_address_and_keeps_original_hostname():
    backend = RecordingBackend(reads=[response_bytes(headers=[(b"Content-Length", b"2")], body=b"ok")])
    transport = HttpCoreTransport(backend_factory=lambda: backend)

    response = await transport.send(
        http_request(), validated=validated_target(), timeout_s=5.0, max_response_bytes=1024
    )

    assert response.status_code == 200
    assert response.body == b"ok"
    assert backend.connect_calls == [ConnectCall(host="93.184.216.34", port=443, timeout=5.0)]
    stream = backend.streams[0]
    assert stream.tls_server_hostnames == ["api.example.com"]
    wire = bytes(stream.written)
    assert b"POST /v1/chat HTTP/1.1\r\n" in wire
    assert b"host: api.example.com\r\n" in wire.lower()
    assert b"93.184.216.34" not in wire


async def test_send_returns_headers_and_applies_timeout_extension():
    backend = RecordingBackend(
        reads=[
            response_bytes(
                headers=[(b"Content-Type", b"application/json"), (b"Content-Length", b"8")],
                body=b'{"ok":1}',
            )
        ]
    )
    transport = HttpCoreTransport(backend_factory=lambda: backend)

    response = await transport.send(
        http_request(), validated=validated_target(), timeout_s=7.5, max_response_bytes=1024
    )

    assert response.headers["content-type"] == "application/json"
    assert response.body == b'{"ok":1}'
    assert backend.connect_calls[0].timeout == 7.5
    stream = backend.streams[0]
    assert stream.read_timeouts
    assert set(stream.read_timeouts) == {7.5}
    assert stream.write_timeouts
    assert set(stream.write_timeouts) == {7.5}


async def test_send_maps_connect_timeout_to_builtin_timeout_error():
    backend = RecordingBackend(reads=[], errors={"93.184.216.34": httpcore.ConnectTimeout("connect timed out")})
    transport = HttpCoreTransport(backend_factory=lambda: backend)

    with pytest.raises(TimeoutError, match="target request timed out"):
        await transport.send(http_request(), validated=validated_target(), timeout_s=5.0, max_response_bytes=1024)


async def test_send_enforces_body_cap_end_to_end():
    # 12 body bytes, delivered as two chunks, against a 10 byte cap.
    backend = RecordingBackend(
        reads=[
            response_bytes(headers=[(b"Content-Length", b"12")]),
            b"aaaa",
            b"bbbbbbbb",
        ]
    )
    transport = HttpCoreTransport(backend_factory=lambda: backend)

    with pytest.raises(TargetResponseTooLargeError, match="response exceeds"):
        await transport.send(http_request(), validated=validated_target(), timeout_s=5.0, max_response_bytes=10)

    assert backend.streams[0].read_timeouts


async def test_send_does_not_follow_redirects():
    backend = RecordingBackend(
        reads=[
            response_bytes(
                status_line=b"HTTP/1.1 302 Found",
                headers=[(b"Location", b"https://evil.example/"), (b"Content-Length", b"0")],
            )
        ]
    )
    transport = HttpCoreTransport(backend_factory=lambda: backend)

    response = await transport.send(
        http_request(), validated=validated_target(), timeout_s=5.0, max_response_bytes=1024
    )

    assert response.status_code == 302
    assert response.headers["location"] == "https://evil.example/"
    assert len(backend.connect_calls) == 1
