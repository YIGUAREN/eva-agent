"""The phase-1 quality expert: prepare rows, run graders, normalize results."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from eva_agent.domain.evidence import EvidenceBundle
from eva_agent.domain.grader import GraderSpec
from eva_agent.domain.plan import Dimension, DimensionPlan
from eva_agent.domain.result import (
    DimensionResult,
    EvidenceRef,
    Finding,
    GraderRunOutput,
    NormalizedQuality,
    Usage,
)
from eva_agent.domain.run import InputKind
from eva_agent.graph.experts.rows import prepare_rows_with_indexes
from eva_agent.tools.graders import RunGradersInput
from eva_agent.tools.registry import ToolRegistry

LOW_SCORE_THRESHOLD = 0.5


class QualityState(TypedDict):
    run_id: str
    plan: DimensionPlan
    evidence: EvidenceBundle
    rows: list[dict[str, str]]
    source_indexes: list[int | None]
    specs: list[GraderSpec]
    outputs: list[GraderRunOutput]
    normalized: NormalizedQuality | None
    result: DimensionResult | None


def _sample_ref(
    evidence: EvidenceBundle,
    row: dict[str, str],
    index: int,
    *,
    source_index: int | None = None,
) -> EvidenceRef:
    excerpt = (row.get("response") or "")[:160] or None
    if evidence.kind is InputKind.DATASET and evidence.dataset is not None:
        return EvidenceRef(
            kind="dataset_row",
            locator=f"{evidence.dataset.path}#{index}",
            excerpt=excerpt,
        )
    if evidence.kind is InputKind.TARGET:
        locator_index = source_index if source_index is not None else index
        return EvidenceRef(kind="target_request", locator=f"target_probe.jsonl#{locator_index}", excerpt=excerpt)
    return EvidenceRef(kind="paste_field", locator="response", excerpt=excerpt)


def _workspace_ref(evidence: EvidenceBundle) -> EvidenceRef:
    if evidence.kind is InputKind.DATASET and evidence.dataset is not None:
        return EvidenceRef(kind="file", locator=evidence.dataset.path)
    if evidence.kind is InputKind.TARGET:
        return EvidenceRef(kind="file", locator="target_probe.jsonl")
    return EvidenceRef(kind="file", locator="paste.json")


def aggregate_outcomes(
    output: GraderRunOutput,
    *,
    specs: list[GraderSpec],
    rows: list[dict[str, str]],
    evidence: EvidenceBundle,
    source_indexes: list[int | None] | None = None,
) -> NormalizedQuality:
    by_sample: dict[int, list[float]] = defaultdict(list)
    by_grader: dict[str, list[float]] = defaultdict(list)
    errors: set[str] = set()
    scored = 0

    for outcome in output.outcomes:
        if outcome.score is None:
            if outcome.error:
                errors.add(outcome.error)
            continue
        scored += 1
        by_sample[outcome.sample_index].append(outcome.score)
        by_grader[outcome.grader].append(outcome.score)

    sample_scores = {index: mean(values) for index, values in sorted(by_sample.items())}
    expected = len(specs) * len(rows)
    confidence = scored / expected if expected else 0.0
    overall = mean(sample_scores.values()) if sample_scores else None

    findings: list[Finding] = []
    for index, score in sample_scores.items():
        if score >= LOW_SCORE_THRESHOLD or index >= len(rows):
            continue
        source_index = source_indexes[index] if source_indexes is not None and index < len(source_indexes) else None
        findings.append(
            Finding(
                id=f"quality-low-{index + 1}",
                severity="medium",
                category="quality",
                title=f"样本 {index + 1} 质量得分偏低",
                description=(f"样本 {index + 1} 的平均质量得分为 {score:.3f}，低于阈值 {LOW_SCORE_THRESHOLD}。"),
                evidence=[_sample_ref(evidence, rows[index], index, source_index=source_index)],
                remediation=("检查该样本的提问与回答内容，确认是否存在答非所问、信息缺失或事实错误。"),
            )
        )

    metrics = {
        "sample_count": len(rows),
        "scored_sample_count": len(sample_scores),
        "expected_outcomes": expected,
        "scored_outcomes": scored,
        "by_grader": {name: round(mean(values), 4) for name, values in by_grader.items()},
        "per_sample": {str(index): round(value, 4) for index, value in sample_scores.items()},
        "errors": sorted(errors),
    }

    return NormalizedQuality(
        score=overall,
        confidence=min(1.0, confidence),
        metrics=metrics,
        findings=findings,
        evidence=[_workspace_ref(evidence)],
        usage=Usage(
            grader_calls=len(output.outcomes),
            samples=len(rows),
            duration_ms=output.duration_ms,
        ),
    )


def summarize_quality(
    normalized: NormalizedQuality,
    *,
    sample_count: int,
    spec_count: int,
) -> DimensionResult:
    scored = int(normalized.metrics.get("scored_sample_count", 0))
    if normalized.score is None:
        status = "failed"
        summary = f"quality 评分全部失败（{scored}/{sample_count} 个样本有可用分数）"
        error = "all grader calls failed"
    elif scored < sample_count or normalized.confidence < 1.0:
        status = "partial"
        summary = f"quality 部分成功：{scored}/{sample_count} 个样本有可用分数，平均 {normalized.score:.3f}"
        error = None
    else:
        status = "ok"
        summary = f"quality 完成：{sample_count} 个样本，平均 {normalized.score:.3f}"
        error = None

    return DimensionResult(
        dimension=Dimension.QUALITY,
        status=status,
        score=normalized.score,
        confidence=normalized.confidence,
        summary=summary,
        metrics={**normalized.metrics, "grader_count": spec_count},
        findings=normalized.findings,
        evidence=normalized.evidence,
        usage=normalized.usage,
        error=error,
    )


def build_quality_expert_graph(tools: ToolRegistry, default_sample_size: int) -> Any:
    async def prepare(state: QualityState) -> dict[str, Any]:
        limit = state["plan"].sample_size or default_sample_size
        rows, source_indexes = await prepare_rows_with_indexes(
            tools,
            state["evidence"],
            run_id=state["run_id"],
            limit=limit,
        )
        if not rows:
            raise ValueError("no rows with a non-empty query and response")
        return {"rows": rows, "source_indexes": source_indexes}

    async def resolve_graders(state: QualityState) -> dict[str, Any]:
        if state["plan"].dimension is not Dimension.QUALITY:
            raise ValueError("quality expert received a non-quality plan")
        specs = [spec for spec in state["plan"].graders if spec.mode == "pointwise"]
        if not specs:
            raise ValueError("quality plan has no pointwise grader")
        return {"specs": specs}

    async def run_graders(state: QualityState) -> dict[str, Any]:
        output = await tools.call(
            "run_graders",
            RunGradersInput(
                specs=state["specs"],
                rows=state["rows"],
                run_id=state["run_id"],
            ),
        )
        return {"outputs": [output]}

    async def normalize(state: QualityState) -> dict[str, Any]:
        outputs = state["outputs"]
        merged = GraderRunOutput(
            outcomes=[outcome for item in outputs for outcome in item.outcomes],
            duration_ms=sum(item.duration_ms for item in outputs),
            error=next((item.error for item in outputs if item.error), None),
        )
        return {
            "normalized": aggregate_outcomes(
                merged,
                specs=state["specs"],
                rows=state["rows"],
                evidence=state["evidence"],
                source_indexes=state["source_indexes"],
            )
        }

    async def summarize(state: QualityState) -> dict[str, Any]:
        normalized = state["normalized"]
        if normalized is None:
            raise ValueError("quality normalization produced no result")
        return {
            "result": summarize_quality(
                normalized,
                sample_count=len(state["rows"]),
                spec_count=len(state["specs"]),
            )
        }

    graph = StateGraph(QualityState)
    graph.add_node("prepare_rows", prepare)
    graph.add_node("resolve_graders", resolve_graders)
    graph.add_node("run_graders", run_graders)
    graph.add_node("normalize", normalize)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "prepare_rows")
    graph.add_edge("prepare_rows", "resolve_graders")
    graph.add_edge("resolve_graders", "run_graders")
    graph.add_edge("run_graders", "normalize")
    graph.add_edge("normalize", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()


class QualityExpert:
    def __init__(self, tools: ToolRegistry, *, default_sample_size: int = 50) -> None:
        self._graph = build_quality_expert_graph(tools, default_sample_size)

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
                "rows": [],
                "source_indexes": [],
                "specs": [],
                "outputs": [],
                "normalized": None,
                "result": None,
            }
        )
        result = final.get("result")
        if result is None:
            raise RuntimeError("quality expert produced no result")
        return result
