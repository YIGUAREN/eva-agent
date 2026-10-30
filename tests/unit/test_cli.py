import json
from pathlib import Path

import pytest

from eva_agent.cli import _exit_code, _view_payload, build_parser, parse_column_map
from eva_agent.domain.run import RunStatus
from eva_agent.runtime.manager import RunStatusView

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


def _view(status: RunStatus) -> RunStatusView:
    return RunStatusView(run_id="run-1", status=status, plan=None, report_json_path=None, errors=[])


def test_exit_code_reports_failed_runs() -> None:
    assert _exit_code(_view(RunStatus.COMPLETED)) == 0
    assert _exit_code(_view(RunStatus.FAILED)) == 1


def test_view_payload_includes_dimension_summary(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "dimensions": [
                    {"dimension": "quality", "status": "ok", "score": 0.8, "summary": "ignored"},
                    {"dimension": "security", "status": "partial", "score": 0.5},
                ]
            }
        ),
        encoding="utf-8",
    )
    view = RunStatusView(
        run_id="run-1",
        status=RunStatus.COMPLETED,
        plan=None,
        report_json_path=str(report),
        errors=[],
    )
    payload = _view_payload(view)
    assert payload["dimensions"] == [
        {"dimension": "quality", "status": "ok", "score": 0.8},
        {"dimension": "security", "status": "partial", "score": 0.5},
    ]
    assert payload["run_id"] == "run-1"
    assert payload["status"] == "completed"
    assert payload["report_json"] == str(report)


def test_view_payload_has_empty_dimensions_without_usable_report(tmp_path: Path) -> None:
    assert _view_payload(_view(RunStatus.COMPLETED))["dimensions"] == []

    missing = RunStatusView(
        run_id="run-1",
        status=RunStatus.COMPLETED,
        plan=None,
        report_json_path=str(tmp_path / "missing.json"),
        errors=[],
    )
    assert _view_payload(missing)["dimensions"] == []

    broken = tmp_path / "broken.json"
    broken.write_text("not json", encoding="utf-8")
    unreadable = RunStatusView(
        run_id="run-1",
        status=RunStatus.COMPLETED,
        plan=None,
        report_json_path=str(broken),
        errors=[],
    )
    assert _view_payload(unreadable)["dimensions"] == []
