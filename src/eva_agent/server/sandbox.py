"""HTTP client for the sandbox-runner extraction service."""

from __future__ import annotations

import string
from typing import Any

import httpx

SAFE_REASONS = frozenset(
    {
        "unsupported_format",
        "corrupt",
        "encrypted",
        "absolute_path",
        "path_traversal",
        "invalid_path",
        "invalid_upload_id",
        "symlink",
        "hardlink",
        "special_file",
        "file_too_large",
        "total_too_large",
        "too_many_files",
        "too_deep",
        "rejected",
    }
)


class SandboxUnavailable(RuntimeError):
    """The sandbox runner could not be reached or returned an unusable response."""


class SandboxRejected(RuntimeError):
    """The sandbox runner rejected the package."""

    def __init__(self, reason: str) -> None:
        safe_reason = reason if reason in SAFE_REASONS else "rejected"
        super().__init__(safe_reason)
        self.reason = safe_reason


def _validate_manifest(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SandboxUnavailable("sandbox returned an invalid manifest")
    archive_format = payload.get("format")
    files = payload.get("files")
    if archive_format not in {"zip", "tar"} or not isinstance(files, list):
        raise SandboxUnavailable("sandbox returned an invalid manifest")
    clean: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise SandboxUnavailable("sandbox returned an invalid manifest")
        path = item.get("path")
        size = item.get("size")
        digest = item.get("sha256")
        if not isinstance(path, str) or not path or not isinstance(size, int) or size < 0:
            raise SandboxUnavailable("sandbox returned an invalid manifest")
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in string.hexdigits for ch in digest):
            raise SandboxUnavailable("sandbox returned an invalid manifest")
        clean.append({"path": path, "size": size, "sha256": digest})
    total = payload.get("total_bytes", 0)
    if not isinstance(total, int) or total < 0:
        raise SandboxUnavailable("sandbox returned an invalid manifest")
    return {"format": archive_format, "files": clean, "total_bytes": total}


class SandboxClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        timeout_s: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._token = token
        self._timeout_s = timeout_s
        self._transport = transport

    async def extract(self, upload_id: str, format: str) -> dict[str, Any]:
        body = {"upload_id": upload_id, "format": format, "relative_path": "package.bin"}
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_s,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    "/extract", json=body, headers={"X-Sandbox-Token": self._token}
                )
        except httpx.HTTPError as exc:
            raise SandboxUnavailable("sandbox runner is unavailable") from exc

        if response.status_code == 401:
            raise SandboxUnavailable("sandbox rejected the shared token")
        if response.status_code == 400:
            try:
                rejection = response.json()
            except ValueError as exc:
                raise SandboxUnavailable("sandbox returned an invalid response") from exc
            detail = rejection.get("detail") if isinstance(rejection, dict) else None
            if (
                not isinstance(detail, dict)
                or detail.get("code") != "upload_rejected"
                or not isinstance(detail.get("message"), str)
            ):
                raise SandboxUnavailable("sandbox returned an invalid response")
            raise SandboxRejected(detail["message"])
        if response.status_code != 200:
            raise SandboxUnavailable("sandbox runner is unavailable")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SandboxUnavailable("sandbox returned an invalid response") from exc
        if not isinstance(payload, dict):
            raise SandboxUnavailable("sandbox returned an invalid response")
        return _validate_manifest(payload.get("manifest"))
