"""Persistence for uploaded evidence packages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.storage.models import UploadRow

UPLOAD_EXTRACTING = "extracting"
UPLOAD_EXTRACTED = "extracted"
UPLOAD_FAILED = "failed"


@dataclass(frozen=True)
class UploadSnapshot:
    id: str
    filename: str
    size_bytes: int
    sha256: str
    format: str
    status: str
    manifest: dict[str, Any] | None
    error: str | None
    created_at: datetime
    expires_at: datetime


def _snapshot(row: UploadRow) -> UploadSnapshot:
    return UploadSnapshot(
        id=row.id,
        filename=row.filename,
        size_bytes=row.size_bytes,
        sha256=row.sha256,
        format=row.format,
        status=row.status,
        manifest=row.manifest_json,
        error=row.error,
        created_at=row.created_at,
        expires_at=row.expires_at,
    )


class UploadRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(
        self,
        upload_id: str,
        *,
        filename: str,
        size_bytes: int,
        sha256: str,
        format: str,
        expires_at: datetime,
    ) -> None:
        async with self._session_factory() as session:
            session.add(
                UploadRow(
                    id=upload_id,
                    filename=filename,
                    size_bytes=size_bytes,
                    sha256=sha256,
                    format=format,
                    status=UPLOAD_EXTRACTING,
                    expires_at=expires_at,
                )
            )
            await session.commit()

    async def get(self, upload_id: str) -> UploadSnapshot | None:
        async with self._session_factory() as session:
            row = await session.get(UploadRow, upload_id)
            return None if row is None else _snapshot(row)

    async def mark_extracted(self, upload_id: str, manifest: dict[str, Any]) -> None:
        await self._update(upload_id, status=UPLOAD_EXTRACTED, manifest_json=manifest, error=None)

    async def mark_failed(self, upload_id: str, error: str) -> None:
        await self._update(upload_id, status=UPLOAD_FAILED, manifest_json=None, error=error[:500])

    async def _update(self, upload_id: str, **values: Any) -> None:
        async with self._session_factory() as session:
            row = await session.get(UploadRow, upload_id)
            if row is None:
                raise KeyError(upload_id)
            for key, value in values.items():
                setattr(row, key, value)
            await session.commit()
