import pytest

from eva_agent.cli import build_parser, parse_column_map

BUDGET_ARGS = [
    "--max-cost-usd",
    "1",
    "--max-wall-clock-s",
    "60",
    "--max-target-requests",
    "1",
]


def test_start_accepts_paste_input() -> None:
    args = build_parser().parse_args(["run", "start", "--query", "q", "--response", "a", *BUDGET_ARGS])

    assert args.command == "run"
    assert args.run_command == "start"
    assert args.query == "q"
    assert args.response == "a"
    assert args.max_target_requests == 1


def test_start_requires_an_input_source() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "start", *BUDGET_ARGS])


def test_start_rejects_two_input_sources() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "start", "--query", "q", "--dataset", "data.csv", *BUDGET_ARGS])


def test_start_requires_every_budget_field() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "start", "--query", "q", "--max-cost-usd", "1"])


def test_dataset_arguments_are_parsed() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "start",
            "--dataset",
            "data/qa.csv",
            "--dataset-format",
            "jsonl",
            "--column-map",
            "response=answer",
            "--column-map",
            "reference=gold",
            *BUDGET_ARGS,
        ]
    )

    assert args.dataset == "data/qa.csv"
    assert args.dataset_format == "jsonl"
    assert args.column_map == ["response=answer", "reference=gold"]


def test_parse_column_map() -> None:
    assert parse_column_map(["response=answer", "reference=gold"]) == {
        "response": "answer",
        "reference": "gold",
    }


def test_parse_column_map_rejects_bad_entry() -> None:
    with pytest.raises(ValueError, match="expected key=column"):
        parse_column_map(["response"])


def test_approve_requires_a_decision() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "approve", "run-1"])


def test_report_command_takes_run_id() -> None:
    args = build_parser().parse_args(["run", "report", "run-1"])

    assert args.run_id == "run-1"
