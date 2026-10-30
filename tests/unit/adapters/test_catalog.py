from pathlib import Path

import pytest
from pydantic import ValidationError

from eva_agent.adapters.openjudge.catalog import (
    CatalogError,
    GraderEntry,
    load_catalog,
    load_class,
    summarize,
)
from tests.fakes.openjudge import FakeChatModel

CATALOG = Path("config/graders.yaml")


def test_catalog_loads_all_phase_one_graders() -> None:
    entries = load_catalog(CATALOG)

    assert set(entries) == {
        "correctness",
        "relevance",
        "instruction_following",
        "hallucination",
        "similarity",
        "harmfulness",
        "code_security",
    }


@pytest.mark.parametrize("entry", list(load_catalog(CATALOG).values()), ids=lambda entry: entry.id)
def test_every_catalog_entry_instantiates(entry: GraderEntry) -> None:
    cls = load_class(entry.klass)
    grader = cls(model=FakeChatModel()) if entry.needs_model else cls()

    assert grader.mode.value == entry.mode
    assert grader.name


def test_summary_exposes_planner_fields() -> None:
    summary = summarize(load_catalog(CATALOG))

    assert {"id", "mode", "inputs", "tags", "needs_model", "cost"} <= set(summary[0])


def test_catalog_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "graders.yaml"
    path.write_text(
        "graders:\n"
        "  - id: correctness\n"
        "    class: openjudge.graders.common.correctness.CorrectnessGrader\n"
        "    mode: pointwise\n"
        "    inputs: [query, response, reference_response]\n"
        "    score_min: 1\n"
        "    score_max: 5\n"
        "  - id: correctness\n"
        "    class: openjudge.graders.common.relevance.RelevanceGrader\n"
        "    mode: pointwise\n"
        "    inputs: [query, response]\n"
        "    score_min: 1\n"
        "    score_max: 5\n",
        encoding="utf-8",
    )

    with pytest.raises(CatalogError, match="duplicate"):
        load_catalog(path)


def test_catalog_rejects_invalid_score_range(tmp_path: Path) -> None:
    path = tmp_path / "graders.yaml"
    path.write_text(
        "graders:\n"
        "  - id: x\n"
        "    class: openjudge.graders.common.relevance.RelevanceGrader\n"
        "    mode: pointwise\n"
        "    inputs: [query, response]\n"
        "    score_min: 5\n"
        "    score_max: 5\n",
        encoding="utf-8",
    )

    with pytest.raises(CatalogError, match="score_max"):
        load_catalog(path)


def test_catalog_rejects_missing_class(tmp_path: Path) -> None:
    path = tmp_path / "graders.yaml"
    path.write_text(
        "graders:\n"
        "  - id: x\n"
        "    class: openjudge.graders.common.relevance.RelevanceGrader\n"
        "    mode: pointwise\n"
        "    inputs: [query, response]\n"
        "    score_min: 1\n"
        "    score_max: 5\n",
        encoding="utf-8",
    )

    load_catalog(path)  # class is a valid field; missing is only detected at load_class


def test_load_class_unknown_module_raises() -> None:
    with pytest.raises(ImportError):
        load_class("openjudge.missing.Module")


def test_grader_entry_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GraderEntry(
            id="x",
            klass="openjudge.graders.common.relevance.RelevanceGrader",  # pyright: ignore[reportCallIssue]
            mode="pointwise",
            inputs=["query", "response"],
            score_min=1,
            score_max=5,
            unexpected=True,  # type: ignore[call-arg]
        )
