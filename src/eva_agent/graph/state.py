"""Shared state for the evaluation graph."""

import operator
from typing import Annotated, TypedDict

from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.run import RunInput, RunStatus


class EvalState(TypedDict):
    run_id: str
    input: RunInput
    budget: Budget
    evidence: EvidenceBundle | None
    plan: EvalPlan | None
    status: RunStatus
    errors: Annotated[list[str], operator.add]
