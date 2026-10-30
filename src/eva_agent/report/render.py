"""Render the run report as JSON and Markdown."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic_core import to_jsonable_python

from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.probe import ProbeSummary
from eva_agent.domain.result import MergedResult, Report
from eva_agent.runtime.artifacts import run_paths
from eva_agent.runtime.probe import ProbeStore

REPORT_VERSION = 3

DEFERRED_CAPABILITIES = [
    "证据包上传与 zip/tar 解压、沙箱执行（阶段 4）",
    "L3 Python grader（v1.5）",
    "预算预扣、结算与 429/5xx 重试（阶段 5）",
    "流式响应、首 token 延迟与并发探测（v1.5）",
]


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _money(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    return value.astimezone(UTC)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def build_report_payload(
    *,
    run_id: str,
    plan: EvalPlan,
    merged: MergedResult,
    generated_at: datetime,
    probe: ProbeSummary | None = None,
) -> dict[str, Any]:
    metrics = to_jsonable_python(merged.metrics)
    return {
        "version": REPORT_VERSION,
        "run_id": run_id,
        "generated_at": _as_utc(generated_at).isoformat(),
        "status": merged.status.value,
        "objective": plan.objective,
        "evidence_summary": plan.evidence_summary,
        "budget": plan.budget.model_dump(mode="json"),
        "probe": probe.model_dump(mode="json") if probe is not None else None,
        "overall_score": metrics.get("overall_score"),
        "dimensions": [item.model_dump(mode="json") for item in merged.results],
        "skipped": [item.model_dump(mode="json") for item in merged.skipped],
        "metrics": metrics,
        "findings": [finding.model_dump(mode="json") for result in merged.results for finding in result.findings],
        "deferred": list(DEFERRED_CAPABILITIES),
    }


def _latency_lines(metrics: dict[str, Any]) -> list[str]:
    return [
        "| count | error_rate | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |",
        "|---|---|---|---|---|---|",
        f"| {_cell(metrics.get('count'))} | {_cell(metrics.get('error_rate'))} | "
        f"{_cell(metrics.get('p50_ms'))} | {_cell(metrics.get('p95_ms'))} | "
        f"{_cell(metrics.get('p99_ms'))} | {_cell(metrics.get('mean_ms'))} |",
        "",
    ]


def _cost_lines(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "| input_tokens | output_tokens | total (USD) | mean (USD) | per 1k (USD) | estimated |",
        "|---|---|---|---|---|---|",
        f"| {_cell(metrics.get('input_tokens_total', 0))} | "
        f"{_cell(metrics.get('output_tokens_total', 0))} | "
        f"{_money(metrics.get('cost_total_usd'))} | {_money(metrics.get('cost_mean_usd'))} | "
        f"{_money(metrics.get('cost_per_1k_requests_usd'))} | "
        f"{_cell(metrics.get('estimated_samples', 0))} |",
        "",
    ]
    estimated = int(metrics.get("estimated_samples") or 0)
    if estimated > 0:
        lines.extend([f"部分 token 为本地估算（estimated_samples={estimated}）。", ""])
    return lines


def _quality_lines(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "**质量评分**",
        "",
        "| grader | mean score | scored samples | errors |",
        "|---|---|---|---|",
    ]
    scored = metrics.get("scored_sample_count", 0)
    errors = len(metrics.get("errors") or [])
    for name, score in sorted((metrics.get("by_grader") or {}).items()):
        lines.append(f"| {_cell(name)} | {_score(score)} | {_cell(scored)} | {_cell(errors)} |")
    lines.append("")
    return lines


def _security_lines(metrics: dict[str, Any]) -> list[str]:
    passive = metrics.get("passive") or {}
    active = metrics.get("active") or {}
    lines = [
        "**被动检查**",
        "",
        "| grader | mean score | samples | errors |",
        "|---|---|---|---|",
    ]
    errors = len(passive.get("errors") or [])
    scored = passive.get("scored_rows", 0)
    for name, score in sorted((passive.get("by_grader") or {}).items()):
        lines.append(f"| {_cell(name)} | {_score(score)} | {_cell(scored)} | {_cell(errors)} |")
    lines.extend(
        [
            "",
            "**主动扫描**",
            "",
            "| status | dataset | prompts | ASR | jailbreak / total | severity |",
            "|---|---|---|---|---|---|",
        ]
    )
    counts = active.get("severity_counts") or {}
    severity = ", ".join(f"{name}={value}" for name, value in sorted(counts.items())) or "-"
    dataset = ", ".join(str(item) for item in (active.get("dataset") or [])) or "-"
    asr = active.get("asr")
    lines.append(
        f"| {_cell(active.get('status'))} | {_cell(dataset)} | {_cell(active.get('num_prompts', 0))} | "
        f"{_score(asr)} | {_cell(active.get('jailbreak', 0))} / {_cell(active.get('total', 0))} | "
        f"{_cell(severity)} |"
    )
    lines.append("")
    if active.get("refused_reason"):
        lines.extend([f"拒绝原因：`{_cell(active['refused_reason'])}`", ""])
    if active.get("degraded"):
        lines.extend(["主动扫描不可用，已降级为被动结果。", ""])
    vulnerabilities = active.get("vulnerabilities") or []
    if vulnerabilities:
        lines.extend(
            [
                "| vulnerability | ASR | severity | jailbreak / total |",
                "|---|---|---|---|",
            ]
        )
        for item in vulnerabilities:
            lines.append(
                f"| {_cell(item.get('vulnerability'))} | {_score(item.get('asr'))} | "
                f"{_cell(item.get('severity'))} | {_cell(item.get('jailbreak'))} / {_cell(item.get('total'))} |"
            )
        lines.append("")
    return lines


def render_markdown(
    payload: dict[str, Any],
    *,
    probe_mean_latency_ms: float | None = None,
) -> str:
    lines = [
        f"# Eva-Agent 评估报告：{payload['run_id']}",
        "",
        "## 概览",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 综合得分：{_score(payload.get('overall_score'))}",
    ]
    failed_dimensions = (payload.get("metrics") or {}).get("failed_dimensions") or []
    if failed_dimensions:
        lines.append(f"- 失败维度：{', '.join(str(item) for item in failed_dimensions)}")
    lines.extend(
        [
            f"- 评估目标：{payload['objective']}",
            f"- 证据摘要：{payload['evidence_summary']}",
            f"- 生成时间：{payload['generated_at']}",
            "",
            "## 计划",
            "",
            f"- 预算：cost ≤ {payload['budget']['max_cost_usd']} 美元，"
            f"时长 ≤ {payload['budget']['max_wall_clock_s']} 秒，"
            f"目标请求 ≤ {payload['budget']['max_target_requests']}",
            "",
            "## 目标探测",
            "",
        ]
    )

    probe = payload.get("probe")
    if probe is None:
        lines.extend(["（本次运行没有目标探测）", ""])
    else:
        lines.extend(
            [
                f"- 记录文件：`{probe['path']}`",
                f"- 总请求：{probe['total']}",
                f"- 成功：{probe['succeeded']}，失败：{probe['failed']}",
            ]
        )
        if probe_mean_latency_ms is not None:
            lines.append(f"- 平均耗时：{probe_mean_latency_ms} ms")
        lines.append("")

    lines.extend(["## 维度结果", ""])
    if not payload["dimensions"]:
        lines.extend(["（没有任何维度产生结果）", ""])
    for result in payload["dimensions"]:
        lines.extend(
            [
                f"### {result['dimension']}",
                "",
                f"- 状态：`{result['status']}`",
                f"- 得分：{_score(result.get('score'))}",
                f"- 置信度：{_score(result.get('confidence'))}",
                f"- 摘要：{result['summary']}",
                f"- grader 调用：{result['usage']['grader_calls']}，"
                f"样本：{result['usage']['samples']}，"
                f"耗时：{result['usage']['duration_ms']} ms",
            ]
        )
        if result.get("error"):
            lines.append(f"- 错误：{result['error']}")
        lines.append("")
        if result["dimension"] == "latency":
            lines.extend(_latency_lines(result.get("metrics") or {}))
        elif result["dimension"] == "cost":
            lines.extend(_cost_lines(result.get("metrics") or {}))
        elif result["dimension"] == "quality":
            lines.extend(_quality_lines(result.get("metrics") or {}))
        elif result["dimension"] == "security":
            lines.extend(_security_lines(result.get("metrics") or {}))

    lines.extend(["## 发现", ""])
    if not payload["findings"]:
        lines.extend(["（未发现问题）", ""])
    else:
        lines.extend(["| ID | 严重度 | 标题 | 证据 |", "|---|---|---|---|"])
        for finding in payload["findings"]:
            evidence = ", ".join(item["locator"] for item in finding["evidence"]) or "-"
            lines.append(
                f"| {_cell(finding['id'])} | {_cell(finding['severity'])} | "
                f"{_cell(finding['title'])} | {_cell(evidence)} |"
            )
        lines.append("")

    lines.extend(["## 未执行维度", ""])
    if not payload["skipped"]:
        lines.extend(["（无）", ""])
    else:
        lines.extend(["| 维度 | 原因 |", "|---|---|"])
        for item in payload["skipped"]:
            lines.append(f"| {_cell(item['dimension'])} | {_cell(item['reason'])} |")
        lines.append("")

    lines.extend(["## 限制", ""])
    lines.extend(f"- {item}" for item in payload["deferred"])
    lines.extend(
        [
            "",
            "## 复现",
            "",
            "```bash",
            f"eva run status {payload['run_id']}",
            f"eva run report {payload['run_id']}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    *,
    data_dir: str | Path,
    run_id: str,
    plan: EvalPlan,
    merged: MergedResult,
    generated_at: datetime,
    probe: ProbeSummary | None = None,
) -> Report:
    generated_at = _as_utc(generated_at)
    paths = run_paths(data_dir, run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    probe_mean_latency_ms = None
    if probe is not None:
        records = ProbeStore(paths.target_probe).read()
        if records:
            probe_mean_latency_ms = round(mean(float(record.latency_ms) for record in records), 2)
    payload = build_report_payload(
        run_id=run_id,
        plan=plan,
        merged=merged,
        probe=probe,
        generated_at=generated_at,
    )
    markdown = render_markdown(payload, probe_mean_latency_ms=probe_mean_latency_ms)
    paths.report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths.report_markdown.write_text(markdown, encoding="utf-8")
    return Report(
        run_id=run_id,
        markdown_path=str(paths.report_markdown),
        json_path=str(paths.report_json),
        generated_at=generated_at,
    )
