"""Reusable run orchestration shared by the CLI and a future FastAPI layer."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.config.settings import AppConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.run import InputKind, RunInput, RunStatus, validate_run_id
from eva_agent.graph.build import build_graph
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.plan_validation import parse_plan_edit
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.dependencies import build_runtime_dependencies
from eva_agent.storage.audit import AuditRepository
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import AuditEventRow
from eva_agent.storage.repository import RunRepository

logger = logging.getLogger(__name__)

ALLOWED_MSGPACK_MODULES: list[tuple[str, str]] = [
    ("eva_agent.domain.budget", "Budget"),
    ("eva_agent.domain.dataset", "DatasetSample"),
    ("eva_agent.domain.evidence", "EvidenceBundle"),
    ("eva_agent.domain.evidence", "EvidenceFile"),
    ("eva_agent.domain.grader", "CheckSpec"),
    ("eva_agent.domain.grader", "GraderSpec"),
    ("eva_agent.domain.grader", "RubricSpec"),
    ("eva_agent.domain.grader", "ScoreSpec"),
    ("eva_agent.domain.plan", "Dimension"),
    ("eva_agent.domain.plan", "DimensionPlan"),
    ("eva_agent.domain.plan", "EvalPlan"),
    ("eva_agent.domain.plan", "SkippedDimension"),
    ("eva_agent.domain.probe", "ProbeRecord"),
    ("eva_agent.domain.probe", "ProbeSummary"),
    ("eva_agent.domain.result", "Approval"),
    ("eva_agent.domain.result", "DimensionResult"),
    ("eva_agent.domain.result", "EvidenceRef"),
    ("eva_agent.domain.result", "Finding"),
    ("eva_agent.domain.result", "GraderRunOutput"),
    ("eva_agent.domain.result", "GraderSampleOutcome"),
    ("eva_agent.domain.result", "MergedResult"),
    ("eva_agent.domain.result", "NormalizedQuality"),
    ("eva_agent.domain.result", "Report"),
    ("eva_agent.domain.result", "Usage"),
    ("eva_agent.domain.run", "DatasetRef"),
    ("eva_agent.domain.run", "InputKind"),
    ("eva_agent.domain.run", "PastedInput"),
    ("eva_agent.domain.run", "RunInput"),
    ("eva_agent.domain.run", "RunStatus"),
    ("eva_agent.domain.target", "TargetKind"),
    ("eva_agent.domain.target", "TargetSpec"),
    ("eva_agent.domain.target", "UsagePaths"),
]


class RunStateError(RuntimeError):
    """Raised when an operation is invalid for the run's current state."""


@dataclass(frozen=True)
class RunStatusView:
    run_id: str
    status: RunStatus
    plan: EvalPlan | None
    report_json_path: str | None
    errors: list[str]


class RunManager:
    def __init__(
        self,
        config: AppConfig,
        *,
        dependencies: GraphDependencies | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._config = config
        self._base_dependencies = dependencies
        self._checkpointer = checkpointer
        self._checkpoint_conn: aiosqlite.Connection | None = None
        self._engine: Any = None
        self._repo: RunRepository | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._audit: AuditRepository | None = None
        self._graph: Any = None
        self._dependencies: GraphDependencies | None = None

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        config: AppConfig,
        *,
        dependencies: GraphDependencies | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> AsyncIterator[RunManager]:
        manager = cls(config, dependencies=dependencies, checkpointer=checkpointer)
        await manager.setup()
        try:
            yield manager
        finally:
            await manager.close()

    async def setup(self) -> None:
        engine = create_engine(self._config.storage.url)
        await init_db(engine)
        self._engine = engine
        session_factory = create_session_factory(engine)
        self._session_factory = session_factory
        self._repo = RunRepository(session_factory)
        audit = AuditRepository(session_factory)
        self._audit = audit

        if self._checkpointer is None:
            path = Path(self._config.storage.checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(path)
            saver = AsyncSqliteSaver(
                conn,
                serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES),
            )
            await saver.setup()
            self._checkpointer = saver
            self._checkpoint_conn = conn

        base = self._base_dependencies or build_runtime_dependencies(self._config, audit=audit)
        self._dependencies = base
        self._graph = build_graph(replace(base, event_sink=self._event_sink), self._checkpointer)

    async def close(self) -> None:
        if self._checkpoint_conn is not None:
            await self._checkpoint_conn.close()
            self._checkpoint_conn = None
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("RunManager.setup() must run before session access")
        return self._session_factory

    async def _event_sink(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._repo is None:
            raise RuntimeError("RunManager.setup() must run before emitting events")
        try:
            await self._repo.append_event(run_id, event_type, payload)
        except Exception:
            logger.warning("failed to persist event '%s' for run '%s'", event_type, run_id, exc_info=True)

    @staticmethod
    def _thread(run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}}

    def _require_repo(self) -> RunRepository:
        if self._repo is None:
            raise RuntimeError("RunManager.setup() must run before repository access")
        return self._repo

    def _require_graph(self) -> Any:
        if self._graph is None:
            raise RuntimeError("RunManager.setup() must run before graph access")
        return self._graph

    def _require_dependencies(self) -> GraphDependencies:
        if self._dependencies is None:
            raise RuntimeError("RunManager.setup() must run before dependency access")
        return self._dependencies

    @staticmethod
    def _validate_plan_edit(
        current: object,
        payload: dict[str, Any],
        catalog_ids: frozenset[str],
        *,
        input_kind: InputKind,
    ) -> None:
        if not isinstance(current, EvalPlan):
            raise RunStateError("run has no planned draft to amend")
        parse_plan_edit(
            payload,
            catalog_ids=set(catalog_ids),
            original=current,
            input_kind=input_kind,
        )

    async def _sync(self, run_id: str, values: dict[str, Any]) -> bool:
        repo = self._require_repo()
        plan = values.get("plan")
        if plan is not None:
            await repo.set_plan(run_id, plan)
        budget = values.get("budget")
        if budget is not None:
            await repo.set_budget(run_id, budget)
        status = values.get("status", RunStatus.RUNNING)
        new_status = status if isinstance(status, RunStatus) else RunStatus(status)
        return await repo.claim(run_id, from_statuses=(RunStatus.RUNNING,), to_status=new_status)

    async def _current_status(self, run_id: str) -> RunStatus:
        repo = self._require_repo()
        row = await repo.get(run_id)
        if row is None:
            raise RunStateError(f"run '{run_id}' no longer exists")
        return RunStatus(row.status)

    @staticmethod
    def _initial_state(run_id: str, run_input: RunInput, budget: Budget) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "input": run_input,
            "budget": budget,
            "evidence": None,
            "plan": None,
            "probe": None,
            "status": RunStatus.QUEUED,
            "errors": [],
            "approval": None,
            "expert_results": [],
            "merged": None,
            "report": None,
        }

    async def create_run(
        self,
        run_input: RunInput,
        budget: Budget,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
        authorization_ref: str | None = None,
        created_by: str | None = None,
    ) -> str:
        repo = self._require_repo()
        dependencies = self._require_dependencies()
        if run_input.kind is InputKind.TARGET and run_input.target_ref not in dependencies.targets:
            raise ValueError(f"unknown target '{run_input.target_ref}'")
        run_id = validate_run_id(run_id or f"run_{uuid4().hex}")
        await repo.create(
            run_id,
            run_input,
            budget,
            idempotency_key=idempotency_key,
            authorization_ref=authorization_ref,
            created_by=created_by,
        )
        await self._event_sink(run_id, "run.created", {"kind": run_input.kind.value})
        return run_id

    async def _invoke_initial(self, run_id: str, run_input: RunInput, budget: Budget) -> None:
        repo = self._require_repo()
        graph = self._require_graph()
        if not await repo.claim(run_id, from_statuses=(RunStatus.QUEUED,), to_status=RunStatus.RUNNING):
            status = await self._current_status(run_id)
            if status is RunStatus.CANCELLED:
                raise RunStateError(f"run '{run_id}' was cancelled")
            if status is RunStatus.RUNNING:
                raise RunStateError(f"run '{run_id}' is already executing")
            if status in (RunStatus.COMPLETED, RunStatus.FAILED):
                raise RunStateError(f"run '{run_id}' is already terminal with status '{status.value}'")
            raise RunStateError(f"run '{run_id}' is not queued (status '{status.value}')")
        try:
            values = await graph.ainvoke(
                self._initial_state(run_id, run_input, budget),
                self._thread(run_id),
            )
        except Exception as exc:
            if await repo.claim(run_id, from_statuses=(RunStatus.RUNNING,), to_status=RunStatus.FAILED):
                await self._event_sink(run_id, "run.failed", {"error": str(exc)})
            raise
        await self._sync(run_id, values)

    async def start(
        self,
        run_input: RunInput,
        budget: Budget,
        *,
        run_id: str | None = None,
    ) -> str:
        created = await self.create_run(run_input, budget, run_id=run_id)
        await self._invoke_initial(created, run_input, budget)
        return created

    async def start_existing(self, run_id: str) -> RunStatusView:
        repo = self._require_repo()
        graph = self._require_graph()
        row = await repo.get(run_id)
        if row is None:
            raise KeyError(run_id)
        snapshot = await graph.aget_state(self._thread(run_id))
        if snapshot.values:
            raise RunStateError(f"run '{run_id}' already has a checkpoint")
        await self._invoke_initial(
            run_id,
            RunInput.model_validate(row.input_json),
            Budget.model_validate(row.budget_json),
        )
        return await self.status(run_id)

    async def resume(self, run_id: str) -> RunStatusView:
        repo = self._require_repo()
        graph = self._require_graph()
        row = await repo.get(run_id)
        if row is None:
            raise KeyError(run_id)
        snapshot = await graph.aget_state(self._thread(run_id))
        if not snapshot.values:
            raise RunStateError(f"run '{run_id}' has no checkpoint to resume")
        if not await repo.claim(run_id, from_statuses=(RunStatus.RUNNING,), to_status=RunStatus.RUNNING):
            status = await self._current_status(run_id)
            if status is RunStatus.CANCELLED:
                raise RunStateError(f"run '{run_id}' was cancelled")
            raise RunStateError(f"run '{run_id}' cannot be resumed from status '{status.value}'")
        try:
            values = await graph.ainvoke(None, self._thread(run_id))
        except Exception as exc:
            if await repo.claim(run_id, from_statuses=(RunStatus.RUNNING,), to_status=RunStatus.FAILED):
                await self._event_sink(run_id, "run.failed", {"error": str(exc)})
            raise
        await self._sync(run_id, values)
        return await self.status(run_id)

    async def cancel(self, run_id: str) -> RunStatusView:
        repo = self._require_repo()
        row = await repo.get(run_id)
        if row is None:
            raise KeyError(run_id)
        status = RunStatus(row.status)
        if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            return await self.status(run_id)
        cancelled = await repo.claim(
            run_id,
            from_statuses=(
                RunStatus.QUEUED,
                RunStatus.RUNNING,
                RunStatus.WAITING_APPROVAL,
                RunStatus.WAITING_BUDGET,
            ),
            to_status=RunStatus.CANCELLED,
        )
        if cancelled:
            await self._event_sink(run_id, "run.cancelled", {})
        return await self.status(run_id)

    async def approve(
        self,
        run_id: str,
        *,
        approved: bool,
        plan: dict[str, Any] | None = None,
    ) -> RunStatusView:
        repo = self._require_repo()
        graph = self._require_graph()
        row = await repo.get(run_id)
        if row is None:
            raise KeyError(run_id)

        snapshot = await graph.aget_state(self._thread(run_id))
        if snapshot.values.get("status") != RunStatus.WAITING_APPROVAL:
            raise RunStateError(f"run '{run_id}' is not waiting for approval")

        decision: dict[str, Any] = {"approved": approved}
        if approved and plan is not None:
            run_input = snapshot.values.get("input")
            if not isinstance(run_input, RunInput):
                raise RunStateError(f"run '{run_id}' has no recorded input")
            self._validate_plan_edit(
                snapshot.values.get("plan"),
                plan,
                self._require_dependencies().catalog_ids,
                input_kind=run_input.kind,
            )
            decision["plan"] = plan
        if not await repo.claim(
            run_id,
            from_statuses=(RunStatus.WAITING_APPROVAL,),
            to_status=RunStatus.RUNNING,
        ):
            status = await self._current_status(run_id)
            if status is RunStatus.CANCELLED:
                raise RunStateError(f"run '{run_id}' was cancelled")
            raise RunStateError(f"run '{run_id}' is not waiting for approval (status '{status.value}')")
        try:
            values = await graph.ainvoke(Command(resume=decision), self._thread(run_id))
        except Exception as exc:
            if await repo.claim(run_id, from_statuses=(RunStatus.RUNNING,), to_status=RunStatus.FAILED):
                await self._event_sink(run_id, "run.failed", {"error": str(exc)})
            raise
        synced = await self._sync(run_id, values)
        if synced and values.get("status") is RunStatus.FAILED:
            await self._event_sink(run_id, "run.failed", {"errors": list(values.get("errors") or [])})
        elif synced:
            await self._event_sink(
                run_id,
                "run.approved" if approved else "run.rejected",
                {"approved": approved},
            )
        return await self.status(run_id)

    async def status(self, run_id: str) -> RunStatusView:
        repo = self._require_repo()
        graph = self._require_graph()
        row = await repo.get(run_id)
        if row is None:
            raise KeyError(run_id)

        snapshot = await graph.aget_state(self._thread(run_id))
        values = snapshot.values if snapshot is not None else {}
        plan = values.get("plan")
        if plan is None and row.plan_json:
            plan = EvalPlan.model_validate(row.plan_json)

        report_path = run_paths(self._config.storage.data_dir, run_id).report_json
        return RunStatusView(
            run_id=run_id,
            status=RunStatus(row.status),
            plan=plan,
            report_json_path=str(report_path) if report_path.is_file() else None,
            errors=list(values.get("errors") or []),
        )

    async def report(self, run_id: str) -> dict[str, Any]:
        repo = self._require_repo()
        row = await repo.get(run_id)
        if row is None:
            raise KeyError(run_id)

        path = run_paths(self._config.storage.data_dir, run_id).report_json
        if not path.is_file():
            raise RunStateError(f"report for run '{run_id}' is not ready")
        return json.loads(path.read_text(encoding="utf-8"))

    async def audit_events(self, run_id: str) -> list[AuditEventRow]:
        if self._audit is None:
            raise RuntimeError("RunManager.setup() must run before audit access")
        return await self._audit.list_for_run(run_id)
