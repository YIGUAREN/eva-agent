"""Planner protocol and the phase-0 deterministic stub."""

from typing import Protocol

from eva_agent.domain.budget import Budget
from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.plan import Dimension, EvalPlan, SkippedDimension


class Planner(Protocol):
    async def plan(self, evidence: EvidenceBundle, budget: Budget) -> EvalPlan: ...


class StubPlanner:
    """Deterministic planner used until the LLM planner lands in phase 1."""

    async def plan(self, evidence: EvidenceBundle, budget: Budget) -> EvalPlan:
        return EvalPlan(
            objective="阶段 0 占位计划：验证工作流连通性",
            evidence_summary=f"kind={evidence.kind.value}, files={len(evidence.files)}",
            dimensions=[],
            skipped=[SkippedDimension(dimension=dimension, reason="阶段 0 未实现该维度") for dimension in Dimension],
            budget=budget,
        )
