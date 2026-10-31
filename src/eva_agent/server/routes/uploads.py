"""Administrator uploads of evidence packages."""

from __future__ import annotations

import hashlib
import shutil
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Request, UploadFile

from eva_agent.runtime.uploads import upload_paths
from eva_agent.sandbox.extract import ExtractionError, detect_format
from eva_agent.server.app import get_state
from eva_agent.server.auth import csrf_protected, current_session
from eva_agent.server.errors import ApiError
from eva_agent.server.sandbox import SandboxRejected, SandboxUnavailable
from eva_agent.storage.models import AdminSessionRow
from eva_agent.storage.uploads import UPLOAD_EXTRACTED, UPLOAD_FAILED

router = APIRouter(prefix="/api/uploads", tags=["uploads"])

_CHUNK = 1024 * 1024
_MAX_FILENAME = 255


class _UploadTooLarge(Exception):
    pass


def _filename(raw: str | None) -> str:
    name = PurePosixPath((raw or "").replace("\\", "/")).name
    if not name or name in {".", ".."}:
        return "package.bin"
    return name[:_MAX_FILENAME]


async def _store(file: UploadFile, target: Path, max_bytes: int) -> tuple[int, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with target.open("wb") as destination:
            while chunk := await file.read(_CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise _UploadTooLarge
                digest.update(chunk)
                destination.write(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if size == 0:
        target.unlink(missing_ok=True)
        raise ApiError(400, "upload_rejected", "上传包为空")
    return size, digest.hexdigest()


async def _extract(state, upload_id: str, archive_format: str) -> None:
    try:
        manifest = await state.sandbox.extract(upload_id, archive_format)
    except SandboxRejected as exc:
        await state.uploads.mark_failed(upload_id, exc.reason)
        raise ApiError(400, "upload_rejected", "上传包未通过安全校验") from None
    except SandboxUnavailable:
        await state.uploads.mark_failed(upload_id, "sandbox_unavailable")
        raise ApiError(503, "unavailable", "解压服务不可用") from None
    except Exception:
        # A malformed sandbox response can raise unexpected errors. Never leave
        # the row stuck in "extracting" or echo the exception text to the client.
        await state.uploads.mark_failed(upload_id, "sandbox_unavailable")
        raise ApiError(503, "unavailable", "解压服务不可用") from None
    await state.uploads.mark_extracted(upload_id, manifest)


@router.post("", status_code=202)
async def create_upload(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    session: Annotated[AdminSessionRow, Depends(csrf_protected)],
) -> dict[str, str]:
    state = get_state(request)
    max_bytes = state.config.web.upload_max_bytes
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes + _CHUNK:
        raise ApiError(413, "upload_rejected", "上传包超过大小限制")

    upload_id = f"upload_{uuid4().hex}"
    paths = upload_paths(state.config.storage.data_dir, upload_id)
    try:
        size, digest = await _store(file, paths.package, max_bytes)
        try:
            archive_format = detect_format(paths.package)
        except ExtractionError:
            raise ApiError(400, "upload_rejected", "仅支持 zip 或 tar 包") from None
        await state.uploads.create(
            upload_id,
            filename=_filename(file.filename),
            size_bytes=size,
            sha256=digest,
            format=archive_format,
            expires_at=state.now() + timedelta(days=state.config.retention_days),
        )
    except _UploadTooLarge:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise ApiError(413, "upload_rejected", "上传包超过大小限制") from None
    except ApiError:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise
    except BaseException:
        shutil.rmtree(paths.root, ignore_errors=True)
        raise
    await _extract(state, upload_id, archive_format)
    return {"upload_id": upload_id, "status": UPLOAD_EXTRACTED}


@router.get("/{upload_id}")
async def get_upload(
    upload_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(current_session)],
) -> dict[str, object]:
    state = get_state(request)
    row = await state.uploads.get(upload_id)
    if row is None:
        raise ApiError(404, "not_found", "上传记录不存在")
    return {
        "upload_id": row.id,
        "filename": row.filename,
        "size_bytes": row.size_bytes,
        "sha256": row.sha256,
        "format": row.format,
        "status": row.status,
        "manifest": row.manifest,
        "error": row.error,
        "created_at": row.created_at.isoformat(),
        "expires_at": row.expires_at.isoformat(),
    }


@router.post("/{upload_id}/retry", status_code=202)
async def retry_upload(
    upload_id: str,
    request: Request,
    session: Annotated[AdminSessionRow, Depends(csrf_protected)],
) -> dict[str, str]:
    state = get_state(request)
    row = await state.uploads.get(upload_id)
    if row is None:
        raise ApiError(404, "not_found", "上传记录不存在")
    if row.status != UPLOAD_FAILED:
        raise ApiError(409, "conflict", "上传任务当前不可重试")
    paths = upload_paths(state.config.storage.data_dir, upload_id)
    if not paths.package.is_file():
        raise ApiError(404, "not_found", "上传包文件不存在")
    try:
        archive_format = detect_format(paths.package)
    except ExtractionError:
        raise ApiError(400, "upload_rejected", "仅支持 zip 或 tar 包") from None
    await _extract(state, upload_id, archive_format)
    return {"upload_id": upload_id, "status": UPLOAD_EXTRACTED}
