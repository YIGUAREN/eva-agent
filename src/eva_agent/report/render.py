"""Render the phase-1 run report as JSON and Markdown."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_core import to_jsonable_python

from eva_agent.domain.plan import EvalPlan
from eva_agent.domain.result import MergedResult, Report
from eva_agent.runtime.artifacts import run_paths

REPORT_VERSION = 1

DEFERRED_CAPABILITIES = [
    "证据包上传与 zip/tar 解压（沙箱阶段实现并验收）",
    "latency、cost、security 专家（阶段 2、阶段 3）",
    "L2 声明式 grader 与 L3 Python grader（阶段 2 / v1.5）",
    "在线目标 API 扫描（阶段 2）",
]


def _score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


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
        "overall_score": metrics.get("overall_score"),
        "dimensions": [item.model_dump(mode="json") for item in merged.results],
        "skipped": [item.model_dump(mode="json") for item in merged.skipped],
        "metrics": metrics,
        "findings": [finding.model_dump(mode="json") for result in merged.results for finding in result.findings],
        "deferred": list(DEFERRED_CAPABILITIES),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Eva-Agent 评估报告：{payload['run_id']}",
        "",
        "## 概览",
        "",
        f"- 状态：`{payload['status']}`",
        f"- 综合得分：{_score(payload.get('overall_score'))}",
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
        "## 维度结果",
        "",
    ]

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
) -> Report:
    generated_at = _as_utc(generated_at)
    paths = run_paths(data_dir, run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload(
        run_id=run_id,
        plan=plan,
        merged=merged,
        generated_at=generated_at,
    )
    markdown = render_markdown(payload)
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
