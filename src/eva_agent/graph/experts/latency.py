"""The phase-2 latency expert: read probe records, score them, summarize."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.result import (
    DimensionResult,
    EvidenceRef,
    Finding,
    GraderRunOutput,
    Usage,
)
from eva_agent.domain.run import InputKind
from eva_agent.tools.declarative import RunDeclarativeChecksInput
from eva_agent.tools.registry import ToolRegistry
from eva_agent.tools.target import ReadTargetProbeInput

PROBE_PATH = "target_probe.jsonl"
MAX_FINDING_EVIDENCE = 5


class LatencyState(TypedDict):
    run_id: str
    plan: DimensionPlan
    evidence: EvidenceBundle
    records: list[ProbeRecord]
    specs: list[GraderSpec]
    outputs: list[GraderRunOutput]
    result: DimensionResult | None


def nearest_rank(values: list[int], percentile: int) -> int | None:
    """Return the nearest-rank percentile; None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def is_successful(record: ProbeRecord) -> bool:
    return record.error is None and record.status_code is not None and 200 <= record.status_code < 300


def build_latency_record(record: ProbeRecord) -> dict[str, Any]:
    return {
        "index": record.index,
        "response": record.response_text,
        "latency_ms": record.latency_ms,
        "status_code": record.status_code,
        "error": record.error,
    }


def _evidence_ref(record: ProbeRecord) -> EvidenceRef:
    status = "none" if record.status_code is None else record.status_code
    excerpt = f"status={status} latency_ms={record.latency_ms}"
    if record.error:
        excerpt = f"{excerpt} error={record.error}"
    return EvidenceRef(
        kind="target_request",
        locator=f"{PROBE_PATH}#{record.index}",
        excerpt=excerpt[:160],
    )


def _failure_reason(record: ProbeRecord) -> str:
    if record.error:
        return record.error
    if record.status_code is None:
        return "no response"
    return f"http {record.status_code}"


def _status_codes(records: list[ProbeRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = "none" if record.status_code is None else str(record.status_code)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _max_latency_threshold(specs: list[GraderSpec]) -> float | None:
    values = [
        check.value
        for spec in specs
        for check in spec.checks or []
        if check.field == "latency_ms" and check.value is not None
    ]
    return max(values) if values else None


def _findings(
    records: list[ProbeRecord],
    failures: list[ProbeRecord],
    successful: list[ProbeRecord],
    specs: list[GraderSpec],
    p95: int | None,
) -> list[Finding]:
    findings: list[Finding] = []
    if failures:
        distribution: dict[str, int] = {}
        for record in failures:
            reason = _failure_reason(record)
            distribution[reason] = distribution.get(reason, 0) + 1
        detail = "，".join(f"{reason} × {count}" for reason, count in sorted(distribution.items()))
        findings.append(
            Finding(
                id="latency-errors",
                severity="medium",
                category="latency",
                title="目标请求存在失败",
                description=f"{len(failures)}/{len(records)} 次目标请求失败：{detail}。",
                evidence=[_evidence_ref(record) for record in failures[:MAX_FINDING_EVIDENCE]],
                remediation="检查目标地址、鉴权与超时配置，并确认目标服务健康。",
            )
        )

    threshold = _max_latency_threshold(specs)
    if threshold is not None and p95 is not None and p95 > threshold:
        slowest = max(successful, key=lambda record: record.latency_ms)
        findings.append(
            Finding(
                id="latency-p95",
                severity="low",
                category="latency",
                title="p95 延迟超过 L2 阈值",
                description=f"p95 为 {p95} ms，超过 L2 延迟阈值 {threshold:g} ms。",
                evidence=[_evidence_ref(slowest)],
                remediation="排查目标服务的排队、上游依赖与超时设置。",
            )
        )
    return findings


def summarize_latency(
    *,
    records: list[ProbeRecord],
    outcomes: list[GraderRunOutput],
    specs: list[GraderSpec],
    sample_size: int,
) -> DimensionResult:
    total = len(records)
    successful = [record for record in records if is_successful(record)]
    failures = [record for record in records if not is_successful(record)]
    latencies = [record.latency_ms for record in successful]

    by_sample: dict[int, list[float]] = defaultdict(list)
    by_grader: dict[str, list[float]] = defaultdict(list)
    errors: set[str] = set()
    scored = 0
    for output in outcomes:
        for outcome in output.outcomes:
            if outcome.score is None:
                if outcome.error:
                    errors.add(outcome.error)
                continue
            scored += 1
            by_sample[outcome.sample_index].append(outcome.score)
            by_grader[outcome.grader].append(outcome.score)

    sample_scores = {index: mean(values) for index, values in sorted(by_sample.items())}
    score = mean(sample_scores.values()) if sample_scores else None
    confidence = min(1.0, scored / sample_size) if sample_size else 0.0
    p50 = nearest_rank(latencies, 50)
    p95 = nearest_rank(latencies, 95)
    p99 = nearest_rank(latencies, 99)

    if not successful or scored == 0:
        status = "failed"
        if not successful:
            error = "no successful target requests"
            summary = f"latency 失败：{total} 次请求均未成功"
        else:
            error = "all declarative checks failed"
            summary = f"latency 失败：{len(successful)} 个成功样本均无 L2 分数"
    elif failures or scored < len(successful):
        status = "partial"
        error = None
        summary = (
            f"latency 部分成功：{len(successful)}/{total} 次请求成功，"
            f"{scored}/{len(successful)} 个成功样本有分数，p95 {p95} ms"
        )
    else:
        status = "ok"
        error = None
        summary = f"latency 完成：{len(successful)}/{total} 次请求成功，p95 {p95} ms"

    metrics = {
        "count": total,
        "success_count": len(successful),
        "failure_count": len(failures),
        "error_rate": round(len(failures) / total, 4) if total else 0.0,
        "status_codes": _status_codes(records),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": round(mean(latencies), 2) if latencies else None,
        "min_ms": min(latencies) if latencies else None,
        "max_ms": max(latencies) if latencies else None,
        "first_token_ms": None,
        "first_token_note": "non-streaming target",
        "grader_scores": {name: round(mean(values), 4) for name, values in by_grader.items()},
        "scored_sample_count": scored,
        "sample_size": sample_size,
        "errors": sorted(errors),
    }

    return DimensionResult(
        dimension=Dimension.LATENCY,
        status=status,
        score=score,
        confidence=confidence,
        summary=summary,
        metrics=metrics,
        findings=_findings(records, failures, successful, specs, p95),
        evidence=[EvidenceRef(kind="target_request", locator=PROBE_PATH)],
        usage=Usage(
            grader_calls=len([item for output in outcomes for item in output.outcomes]),
            samples=total,
            duration_ms=sum(output.duration_ms for output in outcomes),
        ),
        error=error,
    )


def build_latency_expert_graph(tools: ToolRegistry) -> Any:
    async def read_probe(state: LatencyState) -> dict[str, Any]:
        plan = state["plan"]
        evidence = state["evidence"]
        if plan.dimension is not Dimension.LATENCY:
            raise ValueError("latency expert received a non-latency plan")
        if evidence.kind is not InputKind.TARGET or evidence.root is None:
            raise ValueError("latency expert requires target evidence")
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

    async def evaluate(state: LatencyState) -> dict[str, Any]:
        specs = [spec for spec in state["plan"].graders if spec.level == "declarative"]
        if len(specs) != 1:
            raise ValueError("latency plan requires exactly one declarative grader")
        rows = [build_latency_record(record) for record in state["records"] if is_successful(record)]
        if not rows:
            return {"specs": specs, "outputs": []}
        output = await tools.call(
            "run_declarative_checks",
            RunDeclarativeChecksInput(
                specs=specs,
                records=rows,
                run_id=state["run_id"],
            ),
        )
        return {"specs": specs, "outputs": [output]}

    async def summarize(state: LatencyState) -> dict[str, Any]:
        return {
            "result": summarize_latency(
                records=state["records"],
                outcomes=state["outputs"],
                specs=state["specs"],
                sample_size=state["plan"].sample_size or len(state["records"]),
            )
        }

    graph = StateGraph(LatencyState)
    graph.add_node("read_probe", read_probe)
    graph.add_node("evaluate", evaluate)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "read_probe")
    graph.add_edge("read_probe", "evaluate")
    graph.add_edge("evaluate", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


class LatencyExpert:
    def __init__(self, tools: ToolRegistry) -> None:
        self._graph = build_latency_expert_graph(tools)

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
                "outputs": [],
                "result": None,
            }
        )
        result = final.get("result")
        if result is None:
            raise RuntimeError("latency expert produced no result")
        return result
