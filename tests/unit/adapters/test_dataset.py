from pathlib import Path

import pytest

from eva_agent.adapters.inputs.dataset import DatasetError, load_dataset
from eva_agent.domain.run import DatasetRef


def write_csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_canonical_csv(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        "query,response,reference\n首都是哪里,北京是首都,北京\n1+1,2,2\n",
    )

    rows = load_dataset(DatasetRef(path=str(path), format="csv"), max_bytes=10_000, max_rows=10)

    assert rows.columns == ["query", "response", "reference"]
    assert rows.rows == [
        {"query": "首都是哪里", "response": "北京是首都", "reference": "北京"},
        {"query": "1+1", "response": "2", "reference": "2"},
    ]


def test_column_map_renames_source_columns(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "q,a,r\n问题,回答,参考\n")

    rows = load_dataset(
        DatasetRef(
            path=str(path),
            format="csv",
            column_map={"query": "q", "response": "a", "reference": "r"},
        ),
        max_bytes=10_000,
        max_rows=10,
    )

    assert rows.rows == [{"query": "问题", "response": "回答", "reference": "参考"}]


def test_missing_reference_becomes_empty_string(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "query,response\n问题,回答\n")

    rows = load_dataset(DatasetRef(path=str(path), format="csv"), max_bytes=10_000, max_rows=10)

    assert rows.rows == [{"query": "问题", "response": "回答", "reference": ""}]


def test_loads_jsonl_and_unions_columns(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"query": "q1", "response": "a1"}\n\n{"query": "q2", "response": "a2", "reference": "r2"}\n',
        encoding="utf-8",
    )

    rows = load_dataset(DatasetRef(path=str(path), format="jsonl"), max_bytes=10_000, max_rows=10)

    assert rows.columns == ["query", "reference", "response"]
    assert rows.rows[1]["reference"] == "r2"


def test_empty_response_is_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "query,response\n问题,\n")

    with pytest.raises(DatasetError, match="row 1"):
        load_dataset(DatasetRef(path=str(path), format="csv"), max_bytes=10_000, max_rows=10)


def test_missing_mapped_column_is_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "query,response\n问题,回答\n")

    with pytest.raises(DatasetError, match="missing column 'a'"):
        load_dataset(
            DatasetRef(path=str(path), format="csv", column_map={"response": "a"}),
            max_bytes=10_000,
            max_rows=10,
        )


def test_unknown_column_map_target_is_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "query,response\n问题,回答\n")

    with pytest.raises(DatasetError, match="unknown canonical field"):
        load_dataset(
            DatasetRef(path=str(path), format="csv", column_map={"answer": "response"}),
            max_bytes=10_000,
            max_rows=10,
        )


def test_invalid_jsonl_line_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"query": "q", "response": "a"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(DatasetError, match="line 2"):
        load_dataset(DatasetRef(path=str(path), format="jsonl"), max_bytes=10_000, max_rows=10)


def test_row_limit_is_rejected_not_truncated(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "query,response\nq1,a1\nq2,a2\n")

    with pytest.raises(DatasetError, match="max_rows"):
        load_dataset(DatasetRef(path=str(path), format="csv"), max_bytes=10_000, max_rows=1)


def test_byte_limit_is_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "query,response\nq,a\n")

    with pytest.raises(DatasetError, match="max_bytes"):
        load_dataset(DatasetRef(path=str(path), format="csv"), max_bytes=5, max_rows=10)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(DatasetRef(path=str(tmp_path / "nope.csv"), format="csv"), max_bytes=10, max_rows=10)


def test_short_csv_row_is_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "query,response\n问题,回答\n1+1\n")

    with pytest.raises(DatasetError, match="row 2"):
        load_dataset(DatasetRef(path=str(path), format="csv"), max_bytes=10_000, max_rows=10)


def test_extra_csv_values_are_rejected(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "query,response\n问题,回答,多余\n")

    with pytest.raises(DatasetError, match="too many fields"):
        load_dataset(DatasetRef(path=str(path), format="csv"), max_bytes=10_000, max_rows=10)
