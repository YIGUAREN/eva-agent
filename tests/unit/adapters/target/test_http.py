from typing import Any, cast

import pytest
from httpcore import AsyncNetworkBackend, AsyncNetworkStream

from eva_agent.adapters.target.http import (
    PinnedNetworkBackend,
    TargetResponseTooLargeError,
    read_body,
)
from eva_agent.security.target_policy import SecurityPolicyError


class FakeStream:
    pass


class FakeBackend(AsyncNetworkBackend):
    def __init__(self, failures: set[str] | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.failures = failures or set()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        self.calls.append((host, port))
        if host in self.failures:
            raise OSError("connection refused")
        return cast(AsyncNetworkStream, FakeStream())

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
    inner = FakeBackend(failures={"1.1.1.1"})
    backend = PinnedNetworkBackend(host="api.example.com", addresses=("1.1.1.1", "93.184.216.34"), inner=inner)

    await backend.connect_tcp("api.example.com", 443)

    assert inner.calls == [("1.1.1.1", 443), ("93.184.216.34", 443)]


async def test_backend_surfaces_last_connection_error():
    backend = PinnedNetworkBackend(
        host="api.example.com",
        addresses=("1.1.1.1", "93.184.216.34"),
        inner=FakeBackend(failures={"1.1.1.1", "93.184.216.34"}),
    )

    with pytest.raises(OSError, match="connection refused"):
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
