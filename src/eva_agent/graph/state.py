"""Shared state for the evaluation graph."""

import operator
from typing import Annotated, TypedDict

from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import DimensionPlan, EvalPlan
from eva_agent.domain.probe import ProbeSummary
from eva_agent.domain.result import Approval, DimensionResult, MergedResult, Report
from eva_agent.domain.run import RunInput, RunStatus


class EvalState(TypedDict):
    run_id: str
    input: RunInput
    budget: Budget
    evidence: EvidenceBundle | None
    plan: EvalPlan | None
    approval: Approval | None
    status: RunStatus
    errors: Annotated[list[str], operator.add]
    expert_results: Annotated[list[DimensionResult], operator.add]
    merged: MergedResult | None
    report: Report | None


class ExpertState(TypedDict):
    run_id: str
    dimension_plan: DimensionPlan
    evidence: EvidenceBundle
    probe: ProbeSummary | None
    budget: Budget
