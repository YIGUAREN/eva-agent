import pytest
from pydantic import ValidationError

from eva_agent.domain.run import (
    DatasetRef,
    GraderSelection,
    InputKind,
    PastedInput,
    RunInput,
    RunStatus,
    validate_run_id,
)


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


def test_grader_selection_is_absent_by_default() -> None:
    run_input = RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q"))

    assert run_input.graders is None


def test_grader_selection_accepts_catalog_ids_per_dimension() -> None:
    selection = GraderSelection(quality=["relevance"], security=["harmfulness"])

    assert selection.quality == ["relevance"]
    assert selection.security == ["harmfulness"]


@pytest.mark.parametrize("grader_id", ["../escape", "Relevance", "", "a" * 65])
def test_grader_selection_rejects_unsafe_ids(grader_id: str) -> None:
    with pytest.raises(ValidationError):
        GraderSelection(quality=[grader_id])


def test_grader_selection_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError):
        GraderSelection(quality=["relevance", "relevance"])


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


def test_dataset_input_accepts_dataset_ref() -> None:
    ref = DatasetRef(path="data/sample.csv", format="csv", column_map={"response": "answer"})
    run_input = RunInput(kind=InputKind.DATASET, dataset=ref)

    assert run_input.dataset is not None
    assert run_input.dataset.column_map == {"response": "answer"}


def test_dataset_ref_rejects_unknown_format() -> None:
    with pytest.raises(ValidationError):
        DatasetRef(path="data/sample.txt", format="txt")  # type: ignore[arg-type]


def test_dataset_ref_requires_path() -> None:
    with pytest.raises(ValidationError):
        DatasetRef(path="", format="csv")


def test_dataset_payload_rejected_for_paste_input() -> None:
    with pytest.raises(ValidationError):
        RunInput(
            kind=InputKind.PASTE,
            paste=PastedInput(query="q"),
            dataset=DatasetRef(path="data/sample.csv", format="csv"),
        )


@pytest.mark.parametrize("run_id", ["run-1", "run_abc123", "A" * 64])
def test_validate_run_id_accepts_safe_ids(run_id: str) -> None:
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize(
    "run_id",
    ["..", "../../x", ".", "", "a/b", "C:\\x", "a" * 65],
)
def test_validate_run_id_rejects_unsafe_ids(run_id: str) -> None:
    with pytest.raises(ValueError, match="invalid run_id"):
        validate_run_id(run_id)
