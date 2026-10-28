"""Repository for run state and events."""

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.run import RunInput, RunStatus
from eva_agent.storage.models import RunEventCounterRow, RunEventRow, RunRow


class RunRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(self, run_id: str, run_input: RunInput, budget: Budget) -> None:
        async with self._session_factory() as session:
            session.add(
                RunRow(
                    id=run_id,
                    status=RunStatus.QUEUED.value,
                    input_json=run_input.model_dump(mode="json"),
                    budget_json=budget.model_dump(mode="json"),
                )
            )
            await session.flush()
            session.add(RunEventCounterRow(run_id=run_id, next_seq=0))
            await session.commit()

    async def get(self, run_id: str) -> RunRow | None:
        async with self._session_factory() as session:
            return await session.get(RunRow, run_id)

    async def set_status(self, run_id: str, status: RunStatus) -> None:
        async with self._session_factory() as session:
            row = await session.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.status = status.value
            await session.commit()

    async def append_event(self, run_id: str, event_type: str, payload: dict | None = None) -> int:
        async with self._session_factory() as session:
            seq = await session.scalar(
                update(RunEventCounterRow)
                .where(RunEventCounterRow.run_id == run_id)
                .values(next_seq=RunEventCounterRow.next_seq + 1)
                .returning(RunEventCounterRow.next_seq)
            )
            if seq is None:
                raise KeyError(run_id)
            session.add(RunEventRow(run_id=run_id, seq=seq, type=event_type, payload_json=payload or {}))
            await session.commit()
            return seq

    async def set_plan(self, run_id: str, plan: EvalPlan) -> None:
        async with self._session_factory() as session:
            row = await session.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.plan_json = plan.model_dump(mode="json")
            await session.commit()

    async def set_budget(self, run_id: str, budget: Budget) -> None:
        async with self._session_factory() as session:
            row = await session.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.budget_json = budget.model_dump(mode="json")
            await session.commit()
