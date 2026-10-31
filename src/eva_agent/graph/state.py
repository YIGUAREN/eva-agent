"""Shared state for the evaluation graph."""

import operator
from typing import Annotated, NotRequired, TypedDict

from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import Dimension, DimensionPlan, EvalPlan
from eva_agent.domain.probe import ProbeSummary
from eva_agent.domain.result import Approval, DimensionResult, MergedResult, Report
from eva_agent.domain.run import RunInput, RunStatus


def _last_status(current: RunStatus, update: RunStatus) -> RunStatus:
    """Keep the latest status write so parallel experts may all report failures."""
    return update


def merge_expert_results(
    current: list[DimensionResult],
    update: list[DimensionResult],
) -> list[DimensionResult]:
    """Keep the newest result per dimension so an extend retry can supersede a partial."""
    merged: dict[Dimension, DimensionResult] = {result.dimension: result for result in current}
    for result in update:
        merged[result.dimension] = result
    return list(merged.values())


class EvalState(TypedDict):
    run_id: str
    input: RunInput
    budget: Budget
    authorization_ref: NotRequired[str | None]
    requester_is_admin: NotRequired[bool | None]
    evidence: EvidenceBundle | None
    plan: EvalPlan | None
    approval: Approval | None
    probe: ProbeSummary | None
    status: Annotated[RunStatus, _last_status]
    errors: Annotated[list[str], operator.add]
    expert_results: Annotated[list[DimensionResult], merge_expert_results]
    merged: MergedResult | None
    report: Report | None
    finalize_estimate_usd: NotRequired[float]
    finalize_overrun: NotRequired[bool]
    budget_gate: NotRequired[dict[str, object] | None]
    budget_gate_seq: NotRequired[int | None]
    retry_dimensions: NotRequired[list[str]]


class ExpertState(TypedDict):
    """Payload for one expert dispatched through ``Send``.

    The input keys are what an expert node reads. ``status``, ``errors``, and
    ``expert_results`` are the node's output keys, merged back into
    :class:`EvalState` after the parallel fan-out.
    """

    run_id: str
    dimension_plan: DimensionPlan
    evidence: EvidenceBundle
    probe: ProbeSummary | None
    budget: Budget
    authorization_ref: NotRequired[str | None]
    requester_is_admin: NotRequired[bool | None]
    status: NotRequired[RunStatus]
    errors: NotRequired[Annotated[list[str], operator.add]]
    expert_results: NotRequired[Annotated[list[DimensionResult], merge_expert_results]]
