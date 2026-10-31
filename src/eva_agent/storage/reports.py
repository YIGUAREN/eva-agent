"""Persisted report snapshots and downloadable artifacts."""

import hashlib
import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.storage.models import ArtifactRow, ReportRow


@dataclass(frozen=True)
class ReportSnapshot:
    run_id: str
    version: int
    payload: dict
    markdown: str
    sha256: str


class ReportRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert(self, run_id: str, version: int, payload: dict, markdown: str) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        async with self._session_factory() as session:
            row = await session.get(ReportRow, run_id)
            if row is None:
                session.add(ReportRow(run_id=run_id, version=version, json=payload, markdown=markdown, sha256=digest))
            else:
                row.version = version
                row.json = payload
                row.markdown = markdown
                row.sha256 = digest
            await session.commit()

    async def get(self, run_id: str) -> ReportSnapshot | None:
        async with self._session_factory() as session:
            row = await session.get(ReportRow, run_id)
            if row is None:
                return None
            return ReportSnapshot(row.run_id, row.version, row.json, row.markdown, row.sha256)


class ArtifactRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def add(
        self,
        *,
        artifact_id: str,
        run_id: str,
        name: str,
        sha256: str,
        size_bytes: int,
        content_type: str,
        visibility: str,
    ) -> None:
        async with self._session_factory() as session:
            session.add(
                ArtifactRow(
                    id=artifact_id,
                    run_id=run_id,
                    name=name,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    content_type=content_type,
                    visibility=visibility,
                )
            )
            await session.commit()

    async def get(self, run_id: str, name: str) -> ArtifactRow | None:
        async with self._session_factory() as session:
            return await session.scalar(
                select(ArtifactRow).where(ArtifactRow.run_id == run_id, ArtifactRow.name == name)
            )

    async def list_public(self, run_id: str) -> list[ArtifactRow]:
        async with self._session_factory() as session:
            rows = await session.scalars(
                select(ArtifactRow)
                .where(ArtifactRow.run_id == run_id, ArtifactRow.visibility == "public")
                .order_by(ArtifactRow.name)
            )
            return list(rows)
