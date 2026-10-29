import pytest

from eva_agent.cli import build_parser, build_run_input
from eva_agent.domain.run import InputKind


def start_args(*extra: str) -> list[str]:
    return [
        "run",
        "start",
        *extra,
        "--max-cost-usd",
        "1",
        "--max-wall-clock-s",
        "60",
        "--max-target-requests",
        "5",
    ]


def test_target_args_build_target_input() -> None:
    args = build_parser().parse_args(start_args("--target", "demo", "--prompt", "q1", "--prompt", "q2"))

    run_input = build_run_input(args)

    assert run_input.kind is InputKind.TARGET
    assert run_input.target_ref == "demo"
    assert run_input.target_prompts == ["q1", "q2"]


def test_target_requires_prompt() -> None:
    args = build_parser().parse_args(start_args("--target", "demo"))

    with pytest.raises(ValueError, match="--target requires"):
        build_run_input(args)


def test_prompt_requires_target() -> None:
    args = build_parser().parse_args(start_args("--query", "q", "--response", "a", "--prompt", "p"))

    with pytest.raises(ValueError, match="--prompt requires"):
        build_run_input(args)


def test_target_and_query_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(start_args("--target", "demo", "--query", "q"))
