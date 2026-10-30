"""Persistence for privileged-operation audit events."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.domain.audit import AuditEvent
from eva_agent.security.audit import sanitize_detail
from eva_agent.storage.models import AuditEventRow


class AuditRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record(self, event: AuditEvent) -> None:
        async with self._session_factory() as session:
            session.add(
                AuditEventRow(
                    run_id=event.run_id,
                    actor=event.actor,
                    action=event.action,
                    target=event.target,
                    authorization_ref=event.authorization_ref,
                    scan_kind=event.scan_kind,
                    outcome=event.outcome,
                    detail_json=sanitize_detail(event.detail),
                )
            )
            await session.commit()

    async def list_for_run(self, run_id: str) -> list[AuditEventRow]:
        async with self._session_factory() as session:
            result = await session.scalars(
                select(AuditEventRow)
                .where(AuditEventRow.run_id == run_id)
                .order_by(AuditEventRow.id)
            )
            return list(result)
