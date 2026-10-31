"""Shared state for the evaluation graph."""

import operator
from typing import Annotated, NotRequired, TypedDict

from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import DimensionPlan, EvalPlan
from eva_agent.domain.probe import ProbeSummary
from eva_agent.domain.result import Approval, DimensionResult, MergedResult, Report
from eva_agent.domain.run import RunInput, RunStatus


def _last_status(current: RunStatus, update: RunStatus) -> RunStatus:
    """Keep the latest status write so parallel experts may all report failures."""
    return update


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
    expert_results: Annotated[list[DimensionResult], operator.add]
    merged: MergedResult | None
    report: Report | None
    finalize_estimate_usd: NotRequired[float]
    finalize_overrun: NotRequired[bool]
    budget_gate: NotRequired[dict[str, object] | None]
    budget_gate_seq: NotRequired[int | None]


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
    expert_results: NotRequired[Annotated[list[DimensionResult], operator.add]]
