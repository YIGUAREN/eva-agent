"""The phase-2 cost expert: price probe records, score them, summarize."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.result import (
    DimensionResult,
    EvidenceRef,
    GraderRunOutput,
    Usage,
)
from eva_agent.domain.run import InputKind
from eva_agent.graph.experts.aggregation import aggregate_graders
from eva_agent.tools.declarative import RunDeclarativeChecksInput
from eva_agent.tools.pricing import GetPricingInput
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.target import ReadTargetProbeInput

PROBE_PATH = "target_probe.jsonl"


class CostState(TypedDict):
    run_id: str
    plan: DimensionPlan
    evidence: EvidenceBundle
    records: list[ProbeRecord]
    specs: list[GraderSpec]
    rows: list[dict[str, Any]]
    costs: list[Decimal]
    outputs: list[GraderRunOutput]
    result: DimensionResult | None


def build_cost_record(record: ProbeRecord, cost: Decimal) -> dict[str, Any]:
    return {
        "index": record.index,
        "response": record.response_text,
        "latency_ms": record.latency_ms,
        "status_code": record.status_code,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "cost_usd": float(cost),
        "estimated": record.usage_source == "estimated",
    }


def summarize_cost(
    *,
    records: list[ProbeRecord],
    rows: list[dict[str, Any]],
    costs: list[Decimal],
    outcomes: list[GraderRunOutput],
    specs: list[GraderSpec],
    sample_size: int,
) -> DimensionResult:
    total = len(records)
    aggregate = aggregate_graders(outcomes, sample_size)

    total_cost = sum(costs, Decimal("0"))
    mean_cost = total_cost / len(costs) if costs else None
    per_1k = mean_cost * Decimal(1000) if mean_cost is not None else None

    if not rows or aggregate.scored == 0 or aggregate.score == 0:
        status = "failed"
        if not rows:
            error = "no samples with token usage"
            summary = f"cost 失败：{total} 个样本均缺少 token 用量"
        else:
            error = "all declarative checks failed"
            summary = f"cost 失败：{len(rows)} 个计价样本的 L2 检查全部失败"
    elif len(rows) < total or aggregate.scored < len(rows):
        status = "partial"
        error = None
        summary = f"cost 部分成功：{len(rows)}/{total} 个样本计价，总成本 ${total_cost:.6f}"
    else:
        status = "ok"
        error = None
        summary = f"cost 完成：{len(rows)} 个样本计价，总成本 ${total_cost:.6f}"

    input_total = sum(int(row["input_tokens"]) for row in rows)
    output_total = sum(int(row["output_tokens"]) for row in rows)
    estimated = sum(1 for row in rows if row["estimated"])

    metrics = {
        "count": total,
        "priced_samples": len(rows),
        "estimated_samples": estimated,
        "missing_usage_samples": total - len(rows),
        "input_tokens_total": input_total,
        "output_tokens_total": output_total,
        "cost_total_usd": round(float(total_cost), 6),
        "cost_mean_usd": round(float(mean_cost), 6) if mean_cost is not None else None,
        "cost_per_1k_requests_usd": round(float(per_1k), 6) if per_1k is not None else None,
        **aggregate.metrics(sample_size),
    }

    return DimensionResult(
        dimension=Dimension.COST,
        status=status,
        score=aggregate.score,
        confidence=aggregate.confidence,
        summary=summary,
        metrics=metrics,
        findings=[],
        evidence=[EvidenceRef(kind="target_request", locator=PROBE_PATH)],
        usage=Usage(
            grader_calls=aggregate.calls,
            samples=total,
            input_tokens=input_total,
            output_tokens=output_total,
            duration_ms=sum(output.duration_ms for output in outcomes),
        ),
        error=error,
    )


def build_cost_expert_graph(tools: ToolRegistry) -> Any:
    async def read_probe(state: CostState) -> dict[str, Any]:
        plan = state["plan"]
        evidence = state["evidence"]
        if plan.dimension is not Dimension.COST:
            raise ValueError("cost expert received a non-cost plan")
        if evidence.kind is not InputKind.TARGET or evidence.root is None:
            raise ValueError("cost expert requires target evidence")
        # target_probe.jsonl lives under the run root, next to the evidence directory.
        run_root = Path(evidence.root).parent
        output = await tools.call(
            "read_target_probe",
            ReadTargetProbeInput(root=str(run_root), run_id=state["run_id"]),
        )
        limit = plan.sample_size or len(output.records)
        records = output.records[:limit]
        if not records:
            raise ValueError("target probe has no records")
        return {"records": records}

    async def price(state: CostState) -> dict[str, Any]:
        evidence = state["evidence"]
        if evidence.kind is not InputKind.TARGET or evidence.target is None:
            raise ValueError("cost expert requires target evidence")
        price = await tools.call(
            "get_pricing",
            GetPricingInput(
                provider=evidence.target.provider,
                model=evidence.target.model,
                run_id=state["run_id"],
            ),
        )
        rows: list[dict[str, Any]] = []
        costs: list[Decimal] = []
        for record in state["records"]:
            if record.input_tokens is None or record.output_tokens is None:
                continue
            cost = price.cost_usd(
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
            )
            rows.append(build_cost_record(record, cost))
            costs.append(cost)
        return {"rows": rows, "costs": costs}

    async def evaluate(state: CostState) -> dict[str, Any]:
        specs = [spec for spec in state["plan"].graders if spec.level == "declarative"]
        if len(specs) != 1:
            raise ValueError("cost plan requires exactly one declarative grader")
        if not state["rows"]:
            return {"specs": specs, "outputs": []}
        output = await tools.call(
            "run_declarative_checks",
            RunDeclarativeChecksInput(
                specs=specs,
                records=state["rows"],
                run_id=state["run_id"],
            ),
        )
        return {"specs": specs, "outputs": [output]}

    async def summarize(state: CostState) -> dict[str, Any]:
        return {
            "result": summarize_cost(
                records=state["records"],
                rows=state["rows"],
                costs=state["costs"],
                outcomes=state["outputs"],
                specs=state["specs"],
                sample_size=state["plan"].sample_size or len(state["records"]),
            )
        }

    graph = StateGraph(CostState)
    graph.add_node("read_probe", read_probe)
    graph.add_node("price", price)
    graph.add_node("evaluate", evaluate)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "read_probe")
    graph.add_edge("read_probe", "price")
    graph.add_edge("price", "evaluate")
    graph.add_edge("evaluate", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


class CostExpert:
    def __init__(self, tools: ToolRegistry) -> None:
        self._graph = build_cost_expert_graph(tools)

    async def run(
        self,
        *,
        run_id: str,
        plan: DimensionPlan,
        evidence: EvidenceBundle,
    ) -> DimensionResult:
        final = await self._graph.ainvoke(
            {
                "run_id": run_id,
                "plan": plan,
                "evidence": evidence,
                "records": [],
                "specs": [],
                "rows": [],
                "costs": [],
                "outputs": [],
                "result": None,
            }
        )
        result = final.get("result")
        if result is None:
            raise RuntimeError("cost expert produced no result")
        return result
