import pytest
from pydantic import ValidationError

from eva_agent.domain.run import InputKind, PastedInput, RunInput, RunStatus


def test_paste_input_requires_payload() -> None:
    with pytest.raises(ValidationError):
        RunInput(kind=InputKind.PASTE)


def test_upload_input_requires_upload_id() -> None:
    with pytest.raises(ValidationError):
        RunInput(kind=InputKind.UPLOAD)


def test_dataset_input_requires_dataset_id() -> None:
    with pytest.raises(ValidationError):
        RunInput(kind=InputKind.DATASET)


def test_target_input_requires_target_ref() -> None:
    with pytest.raises(ValidationError):
        RunInput(kind=InputKind.TARGET)


def test_valid_paste_input() -> None:
    run_input = RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q", response="a"))

    assert run_input.kind is InputKind.PASTE
    assert run_input.paste is not None
    assert run_input.paste.query == "q"


def test_run_status_values() -> None:
    assert {status.value for status in RunStatus} == {
        "queued",
        "running",
        "waiting_approval",
        "waiting_budget",
        "completed",
        "failed",
        "cancelled",
    }
