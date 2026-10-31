"""Run the shared target probe once, before the target experts fan out."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from eva_agent.budget.context import budget_context
from eva_agent.budget.types import BudgetExhausted
from eva_agent.domain.plan import Dimension, EvalPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.run import InputKind, RunStatus
from eva_agent.graph.dependencies import GraphDependencies
from eva_agent.graph.state import EvalState
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.probe import ProbeStore
from eva_agent.tools.target import CallTargetInput

TARGET_DIMENSIONS = frozenset({Dimension.QUALITY, Dimension.LATENCY, Dimension.COST, Dimension.SECURITY})


def request_count(plan: EvalPlan) -> int:
    sizes: list[int] = []
    for item in plan.dimensions:
        if not item.enabled or item.dimension not in TARGET_DIMENSIONS:
            continue
        if item.sample_size is None or item.sample_size < 1:
            raise ValueError(f"target dimension '{item.dimension.value}' requires a positive sample_size")
        sizes.append(item.sample_size)
    if not sizes:
        raise ValueError("probe requires at least one enabled target dimension")
    return max(sizes)


async def probe_node(state: EvalState, deps: GraphDependencies) -> dict:
    plan = state["plan"]
    evidence = state["evidence"]
    run_input = state["input"]
    if plan is None or evidence is None:
        raise ValueError("probe requires a plan and evidence")
    if run_input.kind is not InputKind.TARGET:
        raise ValueError("probe requires target input")
    if evidence.target is None:
        raise ValueError("probe requires target evidence")
    if not run_input.target_prompts:
        raise ValueError("probe requires at least one prompt")

    count = request_count(plan)
    limit = state["budget"].max_target_requests
    if count > limit:
        raise ValueError(f"probe requests {count} exceed max_target_requests {limit}")

    context = budget_context()
    if context is not None and context.ledger.enforced:
        try:
            await context.ledger.reserve_requests(count)
        except BudgetExhausted:
            snapshot = context.ledger.snapshot()
            await deps.event_sink(
                state["run_id"],
                "budget.exhausted",
                {
                    "spent_usd": float(snapshot.spent_usd),
                    "remaining_usd": float(snapshot.remaining_usd),
                    "dimension": "probe",
                },
            )
            return {"status": RunStatus.FAILED, "errors": ["目标请求预算不足"]}

    paths = run_paths(deps.data_dir, state["run_id"])
    store = ProbeStore(paths.target_probe)
    store.reset()
    await deps.event_sink(
        state["run_id"],
        "probe.started",
        {"target": evidence.target.name, "requests": count},
    )

    prompts = run_input.target_prompts
    for index in range(count):
        prompt = prompts[index % len(prompts)]
        started = time.perf_counter()
        started_at = datetime.now(UTC)
        try:
            record = await deps.tools.call(
                "call_target",
                CallTargetInput(
                    index=index,
                    target=evidence.target,
                    prompt=prompt,
                    run_id=state["run_id"],
                    timeout_s=deps.target_timeout_s,
                    max_response_bytes=deps.max_response_bytes,
                ),
            )
        except Exception as exc:
            # call_target 自身把请求错误写进记录；这里只会拿到工具层的异常。
            record = ProbeRecord(
                index=index,
                prompt=prompt,
                started_at=started_at,
                status_code=None,
                latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
                error=f"{type(exc).__name__}: {exc}",
            )

        store.append(record)
        await deps.event_sink(
            state["run_id"],
            "probe.record",
            {
                "index": record.index,
                "status_code": record.status_code,
                "latency_ms": record.latency_ms,
                "error": record.error,
            },
        )

    summary = store.summary()
    await deps.event_sink(
        state["run_id"],
        "probe.finished",
        {
            "total": summary.total,
            "succeeded": summary.succeeded,
            "failed": summary.failed,
        },
    )
    return {"probe": summary}
