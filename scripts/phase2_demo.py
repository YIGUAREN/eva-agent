"""Run one phase-2 target evaluation and print where the report landed."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from eva_agent.config.settings import load_config
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind, RunInput
from eva_agent.runtime.manager import RunManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-2 Eva-Agent target demo")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--target", default="demo")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--objective", default="评估目标 API 的延迟与成本")
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-wall-clock-s", type=float, required=True)
    parser.add_argument("--max-target-requests", type=int, required=True)
    parser.add_argument("--yes", action="store_true", help="approve the generated plan immediately")
    return parser


async def main(args: argparse.Namespace) -> int:
    load_dotenv()
    config = load_config(Path(args.config))
    run_input = RunInput(
        kind=InputKind.TARGET,
        objective=args.objective,
        target_ref=args.target,
        target_prompts=list(args.prompt),
    )
    budget = Budget(
        max_cost_usd=args.max_cost_usd,
        max_wall_clock_s=args.max_wall_clock_s,
        max_target_requests=args.max_target_requests,
    )

    async with RunManager.open(config) as manager:
        run_id = await manager.start(run_input, budget)
        view = await manager.status(run_id)
        print(f"run_id: {run_id}")
        print(f"status: {view.status.value}")
        if view.plan is not None:
            print(view.plan.model_dump_json(indent=2))
        if not args.yes:
            print(f"review the plan, then: eva run approve {run_id} --approve")
            return 0

        view = await manager.approve(run_id, approved=True)
        print(f"status: {view.status.value}")
        report = await manager.report(run_id)
        print(f"report_json: {view.report_json_path}")

    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(build_parser().parse_args())))
