"""Minimal AI-Infra-Guard task API client."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

import httpx

from eva_agent.adapters.aig.attachments import (
    ATTACHMENT_NAME,
    DOWNLOAD_PATH,
    MAX_ATTACHMENT_BYTES,
    MAX_RESULT_BYTES,
)
from eva_agent.adapters.aig.errors import AigRejected, AigTimeout, AigUnavailable
from eva_agent.config.settings import AigSettings

TASKS_PATH = "/api/v1/app/taskapi/tasks"
STATUS_PATH = "/api/v1/app/taskapi/status/{session_id}"
RESULT_PATH = "/api/v1/app/taskapi/result/{session_id}"
UPLOAD_PATH = "/api/v1/app/taskapi/upload"

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_FILE_URL = re.compile(r"(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]{1,255}")

_SUCCESS_STATUSES = frozenset({"completed", "done"})
_FAILURE_STATUSES = frozenset({"failed", "error", "terminated"})
_IN_PROGRESS_STATUSES = frozenset({"pending", "running", "todo", "doing"})
# AIG's media type for this endpoint has never been observed, so accept anything
# that could be the file and reject only the types that mean "this is an error
# page". The `<html` sniff and the byte cap still apply.
_REJECTED_ATTACHMENT_TYPES = frozenset({"text/html", "application/json", "text/xml", "application/xml"})


class AigScanClient(Protocol):
    async def create(self, body: dict[str, Any]) -> str: ...
    async def wait(self, session_id: str, *, max_wait_s: float) -> str: ...
    async def result(self, session_id: str) -> dict[str, Any]: ...
    async def upload(self, filename: str, content: bytes) -> str: ...
    async def download(self, session_id: str, file_url: str) -> bytes: ...


class AigClient:
    def __init__(self, settings: AigSettings, *, http: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._http = http

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._http is not None:
            yield self._http
            return
        async with httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=self._settings.timeout_s,
        ) as client:
            yield client

    @staticmethod
    def _decode(body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(body)
        except ValueError:
            raise AigUnavailable("AIG returned a non-JSON response") from None
        if not isinstance(payload, dict):
            raise AigUnavailable("AIG returned a non-object response")
        if payload.get("status") != 0:
            # Never include AIG's message: it can echo request values.
            raise AigRejected("AIG rejected the request")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise AigUnavailable("AIG response has no data object")
        return data

    @classmethod
    def _envelope(cls, response: httpx.Response) -> dict[str, Any]:
        if not 200 <= response.status_code < 300:
            raise AigUnavailable(f"AIG returned HTTP {response.status_code}")
        return cls._decode(response.content)

    async def _get_envelope(self, path: str) -> dict[str, Any]:
        """GET a status/result envelope, stopping mid-stream at MAX_RESULT_BYTES."""
        async with self._client() as client:
            async with client.stream("GET", path) as response:
                if not 200 <= response.status_code < 300:
                    raise AigUnavailable(f"AIG returned HTTP {response.status_code}")
                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > MAX_RESULT_BYTES:
                        raise AigUnavailable("AIG returned an oversized result")
        return self._decode(bytes(buffer))

    async def create(self, body: dict[str, Any]) -> str:
        try:
            async with self._client() as client:
                data = self._envelope(await client.post(TASKS_PATH, json=body))
        except (AigRejected, AigUnavailable):
            raise
        except httpx.HTTPError as exc:
            raise AigUnavailable(f"AIG transport failed: {type(exc).__name__}") from None
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
            raise AigUnavailable("AIG response has an invalid session_id")
        return session_id

    async def wait(self, session_id: str, *, max_wait_s: float) -> str:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max_wait_s
        delay = self._settings.poll_initial_s
        while True:
            try:
                data = await self._get_envelope(STATUS_PATH.format(session_id=session_id))
            except (AigRejected, AigUnavailable):
                raise
            except httpx.HTTPError as exc:
                raise AigUnavailable(f"AIG transport failed: {type(exc).__name__}") from None
            status = data.get("status")
            if status in _SUCCESS_STATUSES:
                return status
            if status in _FAILURE_STATUSES:
                raise AigRejected("AIG scan failed")
            if status not in _IN_PROGRESS_STATUSES:
                raise AigRejected("AIG returned an unknown scan status")
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AigTimeout(f"AIG scan {session_id} did not finish in {max_wait_s:.1f}s")
            await asyncio.sleep(min(delay, remaining))
            delay = min(self._settings.poll_max_s, delay * 2)

    async def result(self, session_id: str) -> dict[str, Any]:
        try:
            return await self._get_envelope(RESULT_PATH.format(session_id=session_id))
        except (AigRejected, AigUnavailable):
            raise
        except httpx.HTTPError as exc:
            raise AigUnavailable(f"AIG transport failed: {type(exc).__name__}") from None

    async def upload(self, filename: str, content: bytes) -> str:
        try:
            async with self._client() as client:
                data = self._envelope(
                    await client.post(UPLOAD_PATH, files={"file": (filename, content, "application/zip")})
                )
        except (AigRejected, AigUnavailable):
            raise
        except httpx.HTTPError as exc:
            raise AigUnavailable(f"AIG transport failed: {type(exc).__name__}") from None
        file_url = data.get("fileUrl")
        if not isinstance(file_url, str) or _FILE_URL.fullmatch(file_url) is None:
            raise AigUnavailable("AIG upload response has an invalid fileUrl")
        return file_url

    async def download(self, session_id: str, file_url: str) -> bytes:
        if _SESSION_ID.fullmatch(session_id) is None or ATTACHMENT_NAME.fullmatch(file_url) is None:
            raise AigUnavailable("AIG download request has an invalid name")
        try:
            async with (
                self._client() as client,
                client.stream(
                    "POST",
                    DOWNLOAD_PATH.format(session_id=session_id),
                    json={"fileUrl": file_url},
                ) as response,
            ):
                if not 200 <= response.status_code < 300:
                    raise AigUnavailable(f"AIG returned HTTP {response.status_code}")
                media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if media_type in _REJECTED_ATTACHMENT_TYPES:
                    raise AigUnavailable("AIG returned an invalid attachment type")
                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    # Stop mid-stream so a huge or decompression-bomb body never
                    # has to be buffered in full before the size guard fires.
                    if len(buffer) > MAX_ATTACHMENT_BYTES:
                        raise AigUnavailable("AIG returned an oversized attachment")
        except httpx.HTTPError as exc:
            raise AigUnavailable(f"AIG transport failed: {type(exc).__name__}") from None
        if not buffer:
            raise AigUnavailable("AIG returned an empty attachment")
        if b"<html" in bytes(buffer[:512]).lower():
            raise AigUnavailable("AIG returned an invalid attachment")
        return bytes(buffer)
