"""httpcore transport that pins connections to pre-validated IP addresses."""

from __future__ import annotations

import ssl
from collections.abc import Callable
from typing import Any, cast

import httpcore
from httpcore import AsyncNetworkBackend, AsyncNetworkStream

from eva_agent.adapters.target.base import TargetRequest, TargetResponse
from eva_agent.security.target_policy import SecurityPolicyError, ValidatedTarget


class TargetResponseTooLargeError(ValueError):
    """Raised when the response body exceeds max_response_bytes."""


# httpcore defines AnyIOBackend conditionally when anyio is missing, which hides
# its AsyncNetworkBackend base class from static checkers.
_DEFAULT_BACKEND_FACTORY: Callable[[], AsyncNetworkBackend] = cast(
    Callable[[], AsyncNetworkBackend], httpcore.AnyIOBackend
)


class PinnedNetworkBackend(AsyncNetworkBackend):
    def __init__(self, *, host: str, addresses: tuple[str, ...], inner: AsyncNetworkBackend) -> None:
        self._host = host.lower()
        self._addresses = addresses
        self._inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        if host.lower() != self._host:
            raise SecurityPolicyError(f"unexpected host during connect: {host!r}")

        last_error: OSError | None = None
        for address in self._addresses:
            try:
                return await self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except OSError as exc:
                last_error = exc
        raise last_error or OSError("no pinned address available")

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> AsyncNetworkStream:
        raise SecurityPolicyError("unix sockets are not allowed for targets")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class HttpCoreTransport:
    def __init__(
        self,
        *,
        ssl_context_factory: Callable[[], ssl.SSLContext] = ssl.create_default_context,
        backend_factory: Callable[[], AsyncNetworkBackend] = _DEFAULT_BACKEND_FACTORY,
    ) -> None:
        self._ssl_context_factory = ssl_context_factory
        self._backend_factory = backend_factory

    async def send(
        self,
        request: TargetRequest,
        *,
        validated: ValidatedTarget,
        timeout_s: float,
        max_response_bytes: int,
    ) -> TargetResponse:
        backend = PinnedNetworkBackend(
            host=validated.host,
            addresses=validated.addresses,
            inner=self._backend_factory(),
        )
        pool = httpcore.AsyncConnectionPool(
            ssl_context=self._ssl_context_factory(),
            network_backend=backend,
            http1=True,
            http2=False,
            retries=0,
        )
        extensions = {
            "timeout": {
                "connect": timeout_s,
                "read": timeout_s,
                "write": timeout_s,
                "pool": timeout_s,
            }
        }
        request_headers = [(key.encode("latin-1"), value.encode("latin-1")) for key, value in request.headers.items()]
        try:
            async with (
                pool,
                pool.stream(
                    request.method,
                    request.url,
                    headers=request_headers,
                    content=request.body or None,
                    extensions=extensions,
                ) as response,
            ):
                body = await read_body(response, max_response_bytes)
                headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in response.headers}
                return TargetResponse(status_code=response.status, headers=headers, body=body)
        except httpcore.TimeoutException as exc:
            raise TimeoutError("target request timed out") from exc


async def read_body(response: Any, max_bytes: int) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_stream():
        body.extend(chunk)
        if len(body) > max_bytes:
            raise TargetResponseTooLargeError(f"response exceeds {max_bytes} bytes")
    return bytes(body)
