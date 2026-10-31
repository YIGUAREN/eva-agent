"""Reusable run orchestration shared by the CLI and a future FastAPI layer."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eva_agent.budget.context import BudgetModels, bind_budget
from eva_agent.budget.estimator import CostEstimator
from eva_agent.budget.finalize import estimate_finalize
from eva_agent.budget.ledger import BudgetLedger
from eva_agent.config.settings import AppConfig
from eva_agent.domain.budget import Budget
from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.run import InputKind, RunInput, RunStatus, validate_run_id
from eva_agent.graph.build import build_graph
from eva_agent.graph.dependencies import GateStore, GraphDependencies
from eva_agent.graph.plan_validation import parse_plan_edit
from eva_agent.packages.manifest import PackageManifest
from eva_agent.packages.run_bundle import ImportResult, build_run_bundle, import_run_bundle
from eva_agent.pricing import PricingTable
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.budget_decisions import (
    InvalidBudgetExtension,
    ensure_summarize_allowed,
    validate_extension,
)
from eva_agent.runtime.dependencies import build_runtime_dependencies
from eva_agent.storage.audit import AuditRepository
from eva_agent.storage.budget import (
    BudgetDecisionRepository,
    RunUsageRepository,
    SqlLedgerStore,
)
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.models import AuditEventRow, RunRow
from eva_agent.storage.reports import ArtifactRepository, ReportRepository
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
        enforce_budget: bool = False,
    ) -> None:
        self._config = config
        self._base_dependencies = dependencies
        self._checkpointer = checkpointer
        self._enforce_budget = enforce_budget
        self._checkpoint_conn: aiosqlite.Connection | None = None
        self._engine: Any = None
        self._repo: RunRepository | None = None
        self._reports: ReportRepository | None = None
        self._artifacts: ArtifactRepository | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._audit: AuditRepository | None = None
        self._graph: Any = None
        self._dependencies: GraphDependencies | None = None
        self._estimator: CostEstimator | None = None
        self._models: BudgetModels | None = None
        self._usage_repo: RunUsageRepository | None = None
        self._ledger_store: SqlLedgerStore | None = None
        self._decisions: BudgetDecisionRepository | None = None
        self._ledgers: dict[str, BudgetLedger] = {}
        self._ledger_lock = asyncio.Lock()
        self._decision_locks: dict[str, asyncio.Lock] = {}

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        config: AppConfig,
        *,
        dependencies: GraphDependencies | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        enforce_budget: bool = False,
    ) -> AsyncIterator[RunManager]:
        manager = cls(
            config,
            dependencies=dependencies,
            checkpointer=checkpointer,
            enforce_budget=enforce_budget,
        )
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
        self._reports = ReportRepository(session_factory)
        self._artifacts = ArtifactRepository(session_factory)
        audit = AuditRepository(session_factory)
        self._audit = audit
        self._usage_repo = RunUsageRepository(session_factory)
        self._ledger_store = SqlLedgerStore(self._usage_repo)
        self._decisions = BudgetDecisionRepository(session_factory)
        # Pricing is only needed to enforce a budget; loading it unconditionally
        # would duplicate build_runtime_dependencies and fail an unenforced CLI
        # run on a missing pricing file.  An estimator without a table stays
        # fail-closed, but an unlimited ledger never asks it to price anything.
        pricing = (
            PricingTable.load(Path(self._config.pricing.path)) if self._enforce_budget else None
        )
        self._estimator = CostEstimator(pricing, token_encoding=self._config.target.token_encoding)
        self._models = BudgetModels(
            agent=self._config.models.agent,
            judge=self._config.models.judge,
            attack=self._config.models.attack,
        )

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
        if self._enforce_budget:
            estimator = self._require_estimator()
            judge_model = self._config.models.judge
            decisions = self._decisions

            def finalize_estimator(state: Mapping[str, Any]) -> Decimal:
                return estimate_finalize(
                    estimator,
                    judge_model,
                    plan=state.get("plan"),
                    merged=state.get("merged"),
                    probe=state.get("probe"),
                )

            self._dependencies = replace(
                base,
                event_sink=self._event_sink,
                finalize_estimator=finalize_estimator,
                gate_store=cast(GateStore, decisions),
            )
        else:
            self._dependencies = replace(base, event_sink=self._event_sink)
        self._graph = build_graph(self._dependencies, self._checkpointer)

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

    def _require_estimator(self) -> CostEstimator:
        if self._estimator is None or self._models is None:
            raise RuntimeError("RunManager.setup() must run before budget access")
        return self._estimator

    def _require_models(self) -> BudgetModels:
        if self._models is None:
            raise RuntimeError("RunManager.setup() must run before budget access")
        return self._models

    async def _ledger_for(self, run_id: str) -> BudgetLedger:
        async with self._ledger_lock:
            existing = self._ledgers.get(run_id)
            if existing is not None:
                return existing
            row = await self._require_repo().get(run_id)
            if row is None:
                raise KeyError(run_id)
            state = (
                await self._ledger_store.load(run_id)
                if self._enforce_budget and self._ledger_store is not None
                else None
            )
            ledger = BudgetLedger(
                Budget.model_validate(row.budget_json),
                store=self._ledger_store if self._enforce_budget else None,
                run_id=run_id,
                unlimited=not self._enforce_budget,
                state=state,
            )
            self._ledgers[run_id] = ledger
            return ledger

    @asynccontextmanager
    async def _budget_scope(self, run_id: str) -> AsyncIterator[BudgetLedger]:
        ledger = await self._ledger_for(run_id)
        with bind_budget(ledger, self._require_estimator(), self._require_models()):
            try:
                yield ledger
            finally:
                await ledger.flush()

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

    async def _paused_nodes(self, run_id: str) -> set[str]:
        snapshot = await self._require_graph().aget_state(self._thread(run_id))
        return set(snapshot.next or ())

    async def paused_at_budget_gate(self, run_id: str) -> bool:
        """Return whether the checkpoint is paused at the budget gate.

        The row may still say RUNNING when the process died between the gate
        checkpoint and ``_sync``; callers use this to reconcile that state.
        """
        return "budget_gate" in await self._paused_nodes(run_id)

    async def _has_pending_budget_gate(self, run_id: str) -> bool:
        if self._decisions is None:
            return False
        return await self._decisions.latest_gate(run_id) is not None

    async def _has_queued_work_after_decision(self, run_id: str) -> bool:
        """Return whether a decided run died mid-graph with work still queued.

        Only a RUNNING row with a recorded decision qualifies: ordinary runs
        keep their existing crash semantics, and a checkpoint parked at
        ``budget_gate`` or ``approve`` must stay parked for its own decision.
        """
        if self._decisions is None:
            return False
        if await self._decisions.latest_decision(run_id) is None:
            return False
        row = await self._require_repo().get(run_id)
        if row is None or RunStatus(row.status) is not RunStatus.RUNNING:
            return False
        next_nodes = await self._paused_nodes(run_id)
        return bool(next_nodes) and "budget_gate" not in next_nodes and "approve" not in next_nodes

    async def converge_from_checkpoint(self, run_id: str) -> bool:
        """Reconcile a RUNNING row with the checkpoint when the graph ran ahead.

        A resume (or a gate pause) can finish in the graph before ``_sync``
        persists the row: the process may die between the graph returning and
        the sync, or the sync may fail.  The checkpoint is then the authority,
        and this helper converges the row without invoking the graph again.
        Only RUNNING rows are touched, so a duplicate submission still reports
        the current state instead of resuming.

        Single-writer assumption (Minor 3): one worker process and one active
        command per run, so the compare-and-set claims here cannot race a
        second writer.  Returns True when the row changed.
        """
        repo = self._require_repo()
        row = await repo.get(run_id)
        if row is None or RunStatus(row.status) is not RunStatus.RUNNING:
            return False
        snapshot = await self._require_graph().aget_state(self._thread(run_id))
        if not snapshot.values:
            return False
        next_nodes = set(snapshot.next or ())
        if "budget_gate" in next_nodes:
            # Park the row only when the gate is still pending.  A recorded
            # decision means the row must stay RUNNING so the replay path can
            # finish the run (Minor 2).
            if not await self._has_pending_budget_gate(run_id):
                return False
            return await repo.claim(
                run_id,
                from_statuses=(RunStatus.RUNNING,),
                to_status=RunStatus.WAITING_BUDGET,
            )
        if "approve" in next_nodes:
            return await repo.claim(
                run_id,
                from_statuses=(RunStatus.RUNNING,),
                to_status=RunStatus.WAITING_APPROVAL,
            )
        if next_nodes:
            # Still mid-graph with work queued: only a real resume can advance it.
            return False
        # The graph reached END, so the checkpoint values carry the final state.
        return await self._sync(run_id, snapshot.values)

    async def _sync(self, run_id: str, values: dict[str, Any]) -> bool:
        try:
            return await self._sync_values(run_id, values)
        except Exception:
            # The graph may already be paused at the gate even though persisting
            # the rest of the state failed.  Reconcile the row to the checkpoint
            # so a later decision retry can recover the run instead of wedging it.
            try:
                paused = await self._paused_nodes(run_id)
            except Exception:
                paused = set()
            if "budget_gate" in paused:
                await self._require_repo().claim(
                    run_id,
                    from_statuses=(RunStatus.RUNNING,),
                    to_status=RunStatus.WAITING_BUDGET,
                )
            raise

    async def _sync_values(self, run_id: str, values: dict[str, Any]) -> bool:
        repo = self._require_repo()
        plan = values.get("plan")
        if plan is not None:
            await repo.set_plan(run_id, plan)
        budget = values.get("budget")
        if budget is not None:
            await repo.set_budget(run_id, budget)
        status = values.get("status", RunStatus.RUNNING)
        new_status = status if isinstance(status, RunStatus) else RunStatus(status)
        # The paused node is the authority: a gate prepare forces RUNNING, so the
        # waiting state must come from the checkpoint rather than the values.
        next_nodes = await self._paused_nodes(run_id)
        if "budget_gate" in next_nodes:
            new_status = RunStatus.WAITING_BUDGET
        elif "approve" in next_nodes:
            new_status = RunStatus.WAITING_APPROVAL
        return await repo.claim(run_id, from_statuses=(RunStatus.RUNNING,), to_status=new_status)

    async def _current_status(self, run_id: str) -> RunStatus:
        repo = self._require_repo()
        row = await repo.get(run_id)
        if row is None:
            raise RunStateError(f"run '{run_id}' no longer exists")
        return RunStatus(row.status)

    @staticmethod
    def _initial_state(
        run_id: str,
        run_input: RunInput,
        budget: Budget,
        *,
        authorization_ref: str | None = None,
        requester_is_admin: bool | None = None,
    ) -> dict[str, Any]:
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
            "authorization_ref": authorization_ref,
            "requester_is_admin": requester_is_admin,
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

    async def _invoke_initial(
        self,
        run_id: str,
        run_input: RunInput,
        budget: Budget,
        *,
        authorization_ref: str | None = None,
        requester_is_admin: bool | None = None,
    ) -> None:
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
            async with self._budget_scope(run_id):
                values = await graph.ainvoke(
                    self._initial_state(
                        run_id,
                        run_input,
                        budget,
                        authorization_ref=authorization_ref,
                        requester_is_admin=requester_is_admin,
                    ),
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
        authorization_ref: str | None = None,
        created_by: str | None = None,
    ) -> str:
        created = await self.create_run(
            run_input,
            budget,
            run_id=run_id,
            authorization_ref=authorization_ref,
            created_by=created_by,
        )
        await self._invoke_initial(
            created,
            run_input,
            budget,
            authorization_ref=authorization_ref,
            requester_is_admin=True if created_by == "admin" else None,
        )
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
            authorization_ref=row.authorization_ref,
            requester_is_admin=True if row.created_by == "admin" else None,
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
        if "budget_gate" in set(snapshot.next or ()):
            # Only decide_budget may continue a run paused at the budget gate.
            raise RunStateError(f"run '{run_id}' is waiting for a budget decision")
        if not await repo.claim(run_id, from_statuses=(RunStatus.RUNNING,), to_status=RunStatus.RUNNING):
            status = await self._current_status(run_id)
            if status is RunStatus.CANCELLED:
                raise RunStateError(f"run '{run_id}' was cancelled")
            raise RunStateError(f"run '{run_id}' cannot be resumed from status '{status.value}'")
        try:
            async with self._budget_scope(run_id):
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
            async with self._budget_scope(run_id):
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

    def _decision_lock_for(self, run_id: str) -> asyncio.Lock:
        lock = self._decision_locks.get(run_id)
        if lock is None:
            lock = asyncio.Lock()
            self._decision_locks[run_id] = lock
        return lock

    async def decide_budget(
        self,
        run_id: str,
        *,
        action: str,
        confirm_overrun: bool = False,
        budget: Budget | None = None,
    ) -> RunStatusView:
        # Two deliveries for the same run must never both replay the recorded
        # decision: the first resumes the graph, the second observes the result.
        async with self._decision_lock_for(run_id):
            return await self._decide_budget_locked(
                run_id,
                action=action,
                confirm_overrun=confirm_overrun,
                budget=budget,
            )

    async def _decide_budget_locked(
        self,
        run_id: str,
        *,
        action: str,
        confirm_overrun: bool,
        budget: Budget | None,
    ) -> RunStatusView:
        repo = self._require_repo()
        row = await repo.get(run_id)
        if row is None:
            raise KeyError(run_id)
        if self._decisions is None:
            raise RunStateError("budget decisions are unavailable without enforcement")

        paused_at_gate = await self.paused_at_budget_gate(run_id)
        gate_row = await self._decisions.latest_gate(run_id)
        if gate_row is None:
            if paused_at_gate:
                # The decision row was rewritten but the process died before the
                # graph resumed; replay the recorded decision instead of treating
                # the missing gate as "already resumed".
                return await self._replay_recorded_decision(run_id)
            if await self._has_queued_work_after_decision(run_id):
                # The decision resumed the graph, which then died mid-flight
                # (merge/analyze/report still queued).  Reuse the ordinary
                # resume path under this decision lock so the run finishes
                # instead of reporting RUNNING forever.  A resume failure
                # propagates, leaving the command un-done for a later retry.
                return await self.resume(run_id)
            # A duplicate delivery after a successful resume is idempotent, but
            # the first delivery may have finished the graph without persisting
            # the row (the process died or _sync failed).  Converge from the
            # checkpoint before reporting the current state.
            await self.converge_from_checkpoint(run_id)
            return await self.status(run_id)
        if not paused_at_gate:
            raise RunStateError(f"run '{run_id}' is not waiting for a budget decision")

        status = RunStatus(row.status)
        if status is RunStatus.RUNNING:
            # The gate checkpoint landed but the row was never synced.  Treat the
            # checkpoint as the authority and make the waiting state durable so
            # the ordinary claim below succeeds.
            if not await repo.claim(
                run_id,
                from_statuses=(RunStatus.RUNNING,),
                to_status=RunStatus.WAITING_BUDGET,
            ):
                return await self.status(run_id)
        elif status is not RunStatus.WAITING_BUDGET:
            raise RunStateError(f"run '{run_id}' is not waiting for a budget decision")

        seq = int(gate_row.seq)
        gate = dict(gate_row.payload_json)
        finalize_estimate = Decimal(str(gate["finalize_estimate"]))
        remaining = Decimal(str(gate["remaining"]))
        usage = (
            RunUsageRepository.to_usage(await self._usage_repo.get(run_id))
            if self._usage_repo is not None
            else None
        )

        decision: dict[str, Any]
        if action == "summarize":
            ensure_summarize_allowed(
                finalize_estimate=finalize_estimate,
                remaining=remaining,
                confirm_overrun=confirm_overrun,
            )
            decision = {"action": "summarize", "confirm_overrun": bool(confirm_overrun)}
        elif action == "extend":
            if budget is None or usage is None:
                raise InvalidBudgetExtension("max_cost_usd", "extend requires a budget")
            validate_extension(budget, usage)
            decision = {"action": "extend", "budget": budget.model_dump(mode="json")}
        else:
            raise ValueError(f"unknown budget decision: {action!r}")

        # Claim, then record, then resume: a duplicate delivery loses the claim
        # and reports the current state without touching the graph.
        claimed = await repo.claim(
            run_id,
            from_statuses=(RunStatus.WAITING_BUDGET,),
            to_status=RunStatus.RUNNING,
        )
        if not claimed:
            current = await repo.get(run_id)
            if current is not None and current.status == RunStatus.RUNNING.value:
                return await self.status(run_id)
            raise RunStateError(f"run '{run_id}' is not waiting for a budget decision")
        recorded = await self._decisions.decide(
            run_id,
            seq,
            action,
            {**gate, "decision": decision},
        )
        if not recorded:
            # Another delivery already decided this gate.  Never leave the row
            # claimed RUNNING with no resume: replay the stored decision when the
            # checkpoint is still parked, otherwise converge from the checkpoint,
            # and finally release the claim so a later retry can finish the run.
            if (
                await self._decisions.latest_decision(run_id) is not None
                and await self.paused_at_budget_gate(run_id)
            ):
                return await self._replay_recorded_decision(run_id)
            if not await self.converge_from_checkpoint(run_id):
                await repo.claim(
                    run_id,
                    from_statuses=(RunStatus.RUNNING,),
                    to_status=RunStatus.WAITING_BUDGET,
                )
            return await self.status(run_id)
        await self._apply_recorded_extension(run_id, decision)
        values = await self._invoke_resume(run_id, decision)
        await self._sync(run_id, values)
        return await self.status(run_id)

    async def _resume_recorded_decision(self, run_id: str) -> RunStatusView:
        async with self._decision_lock_for(run_id):
            return await self._replay_recorded_decision(run_id)

    async def _replay_recorded_decision(self, run_id: str) -> RunStatusView:
        if self._decisions is None:
            raise RunStateError("budget decisions are unavailable without enforcement")
        repo = self._require_repo()
        row = await repo.get(run_id)
        if row is None:
            raise KeyError(run_id)
        decided = await self._decisions.latest_decision(run_id)
        if decided is None:
            raise RunStateError(f"run '{run_id}' is waiting for a budget decision")
        # The row is the durable authority: only a RUNNING run can be waiting on
        # a crashed replay.  Anything else already resumed and is idempotent.
        if RunStatus(row.status) is not RunStatus.RUNNING:
            return await self.status(run_id)
        if not await self.paused_at_budget_gate(run_id):
            return await self.status(run_id)
        stored = decided.payload_json or {}
        decision = dict(stored.get("decision") or {})
        if not decision:
            raise RunStateError(f"run '{run_id}' has no recorded budget decision")
        await self._apply_recorded_extension(run_id, decision)
        values = await self._invoke_resume(run_id, decision)
        await self._sync(run_id, values)
        return await self.status(run_id)

    async def _apply_recorded_extension(self, run_id: str, decision: dict[str, Any]) -> None:
        if decision.get("action") != "extend":
            return
        extended = Budget.model_validate(decision.get("budget"))
        await self._require_repo().set_budget(run_id, extended)
        (await self._ledger_for(run_id)).update_budget(extended)

    async def _invoke_resume(self, run_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        repo = self._require_repo()
        graph = self._require_graph()
        try:
            async with self._budget_scope(run_id):
                values = await graph.ainvoke(Command(resume=decision), self._thread(run_id))
        except Exception as exc:
            if await repo.claim(run_id, from_statuses=(RunStatus.RUNNING,), to_status=RunStatus.FAILED):
                await self._event_sink(run_id, "run.failed", {"error": str(exc)})
            raise
        return values

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

    async def run_row(self, run_id: str) -> RunRow:
        """The persisted row for *run_id*, raising ``KeyError`` when it is missing."""
        row = await self._require_repo().get(run_id)
        if row is None:
            raise KeyError(run_id)
        return row

    async def report(self, run_id: str) -> dict[str, Any]:
        repo = self._require_repo()
        row = await repo.get(run_id)
        if row is None:
            raise KeyError(run_id)

        path = run_paths(self._config.storage.data_dir, run_id).report_json
        if not path.is_file():
            raise RunStateError(f"report for run '{run_id}' is not ready")
        return json.loads(path.read_text(encoding="utf-8"))

    async def export_run_bundle(self, run_id: str, out_dir: Path) -> tuple[Path, PackageManifest]:
        """Build a public run bundle for *run_id* under *out_dir*."""
        async with self.session_factory() as session:
            run_row = await session.get(RunRow, run_id)
        if run_row is None:
            raise KeyError(run_id)
        if self._reports is None or self._artifacts is None:
            raise RuntimeError("RunManager.setup() must run before export")
        report = await self._reports.get(run_id)
        artifacts = await self._artifacts.list_public(run_id)
        return await build_run_bundle(
            config=self._config,
            run_row=run_row,
            report=report,
            artifacts=artifacts,
            out_dir=out_dir,
        )

    async def import_run_bundle(self, archive: Path, *, public: bool) -> ImportResult:
        """Restore a run bundle, skipping an existing ``run_id`` without overwriting it."""
        return await import_run_bundle(config=self._config, archive=archive, public=public)

    async def audit_events(self, run_id: str) -> list[AuditEventRow]:
        if self._audit is None:
            raise RuntimeError("RunManager.setup() must run before audit access")
        return await self._audit.list_for_run(run_id)
