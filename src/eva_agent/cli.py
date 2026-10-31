"""Minimal internal CLI; the interview demo uses the report files, not this CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from eva_agent.config.settings import AppConfig, ConfigError, MissingSecretError, load_config
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import DatasetRef, InputKind, PastedInput, RunInput, RunStatus
from eva_agent.packages.grader_package import build_grader_package, read_grader_package
from eva_agent.packages.manifest import sha256_file
from eva_agent.packages.replay import replay_package
from eva_agent.runtime.manager import RunManager, RunStateError, RunStatusView


def parse_column_map(items: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ValueError(f"invalid column map entry '{item}'; expected key=column")
        mapping[key.strip()] = value.strip()
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eva")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run_subparsers = run.add_subparsers(dest="run_command", required=True)

    start = run_subparsers.add_parser("start")
    inputs = start.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--query")
    inputs.add_argument("--dataset")
    inputs.add_argument("--target")
    start.add_argument("--prompt", action="append", default=[])
    start.add_argument("--response")
    start.add_argument("--context")
    start.add_argument("--dataset-format", choices=["csv", "jsonl"], default="csv")
    start.add_argument("--column-map", action="append", default=[])
    start.add_argument("--objective")
    start.add_argument("--max-cost-usd", type=float, required=True)
    start.add_argument("--max-wall-clock-s", type=float, required=True)
    start.add_argument("--max-target-requests", type=int, required=True)
    start.add_argument("--config", default="config/config.yaml")

    approve = run_subparsers.add_parser("approve")
    approve.add_argument("run_id")
    decision = approve.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    approve.add_argument("--plan-file")
    approve.add_argument("--config", default="config/config.yaml")

    status = run_subparsers.add_parser("status")
    status.add_argument("run_id")
    status.add_argument("--config", default="config/config.yaml")

    report = run_subparsers.add_parser("report")
    report.add_argument("run_id")
    report.add_argument("--config", default="config/config.yaml")

    export_package = subparsers.add_parser("export-grader-package")
    export_package.add_argument("run_id")
    export_package.add_argument("--out", required=True)
    export_package.add_argument("--config", default="config/config.yaml")

    replay = subparsers.add_parser("replay")
    replay.add_argument("package")
    replay.add_argument("--out", required=True)
    return parser


def _dimensions_summary(report_path: str | None) -> list[dict[str, Any]]:
    if not report_path:
        return []
    try:
        raw = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, dict):
        return []
    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, list):
        return []
    summary: list[dict[str, Any]] = []
    for item in dimensions:
        if not isinstance(item, dict):
            continue
        summary.append(
            {
                "dimension": item.get("dimension"),
                "status": item.get("status"),
                "score": item.get("score"),
            }
        )
    return summary


def _view_payload(view: RunStatusView) -> dict[str, Any]:
    return {
        "run_id": view.run_id,
        "status": view.status.value,
        "plan": view.plan.model_dump(mode="json") if view.plan is not None else None,
        "report_json": view.report_json_path,
        "dimensions": _dimensions_summary(view.report_json_path),
        "errors": view.errors,
    }


def _exit_code(view: RunStatusView) -> int:
    return 1 if view.status is RunStatus.FAILED else 0


def build_run_input(args: argparse.Namespace) -> RunInput:
    if args.target:
        if not args.prompt:
            raise ValueError("--target requires at least one --prompt")
        return RunInput(
            kind=InputKind.TARGET,
            objective=args.objective,
            target_ref=args.target,
            target_prompts=list(args.prompt),
        )
    if args.prompt:
        raise ValueError("--prompt requires --target")
    if args.dataset:
        return RunInput(
            kind=InputKind.DATASET,
            objective=args.objective,
            dataset=DatasetRef(
                path=args.dataset,
                format=args.dataset_format,
                column_map=parse_column_map(args.column_map),
            ),
        )
    return RunInput(
        kind=InputKind.PASTE,
        objective=args.objective,
        paste=PastedInput(
            query=args.query,
            response=args.response,
            context=args.context,
        ),
    )


def _load_plan_file(path: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"plan file must contain a mapping: {path}")
    plan = raw.get("plan", raw)
    if not isinstance(plan, dict):
        raise ValueError(f"plan file does not contain a plan mapping: {path}")
    return plan


async def _start(args: argparse.Namespace, config: AppConfig) -> int:
    run_input = build_run_input(args)
    budget = Budget(
        max_cost_usd=args.max_cost_usd,
        max_wall_clock_s=args.max_wall_clock_s,
        max_target_requests=args.max_target_requests,
    )
    async with RunManager.open(config, enforce_budget=False) as manager:
        run_id = await manager.start(run_input, budget)
        view = await manager.status(run_id)
    print(json.dumps(_view_payload(view), ensure_ascii=False, indent=2))
    return _exit_code(view)


async def _approve(args: argparse.Namespace, config: AppConfig) -> int:
    plan = _load_plan_file(args.plan_file) if args.plan_file else None
    if plan is not None and args.reject:
        raise ValueError("cannot edit a plan while rejecting it")
    async with RunManager.open(config, enforce_budget=False) as manager:
        view = await manager.approve(args.run_id, approved=args.approve, plan=plan)
    print(json.dumps(_view_payload(view), ensure_ascii=False, indent=2))
    return _exit_code(view)


async def _status(args: argparse.Namespace, config: AppConfig) -> int:
    async with RunManager.open(config, enforce_budget=False) as manager:
        view = await manager.status(args.run_id)
    print(json.dumps(_view_payload(view), ensure_ascii=False, indent=2))
    return 0


async def _report(args: argparse.Namespace, config: AppConfig) -> int:
    async with RunManager.open(config, enforce_budget=False) as manager:
        payload = await manager.report(args.run_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


async def _export_grader_package(args: argparse.Namespace, config: AppConfig) -> int:
    async with RunManager.open(config, enforce_budget=False) as manager:
        run_row = await manager.run_row(args.run_id)
        archive, manifest = await build_grader_package(
            config=config, run_row=run_row, out_dir=Path(args.out)
        )
    print(
        json.dumps(
            {
                "package": str(archive),
                "sha256": sha256_file(archive),
                "size_bytes": archive.stat().st_size,
                "replayable": manifest.replayable,
                "replay_blockers": manifest.replay_blockers,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


async def _replay(args: argparse.Namespace, _config: AppConfig | None) -> int:
    package = read_grader_package(Path(args.package))
    result = replay_package(package)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "matched": result.matched,
        "compared": result.compared,
        "skipped": result.skipped,
        "differences": [
            {"path": item.path, "recorded": item.recorded, "replayed": item.replayed}
            for item in result.differences
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    (out_dir / "replay.json").write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0 if result.matched else 1


async def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "replay":
        return await _replay(args, None)
    config = load_config(Path(args.config))
    if args.command == "export-grader-package":
        return await _export_grader_package(args, config)
    if args.run_command == "start":
        return await _start(args, config)
    if args.run_command == "approve":
        return await _approve(args, config)
    if args.run_command == "status":
        return await _status(args, config)
    return await _report(args, config)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_dispatch(args))
    except KeyError as exc:
        print(f"error: unknown run {exc}", file=sys.stderr)
        return 1
    except (ConfigError, MissingSecretError, RunStateError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
