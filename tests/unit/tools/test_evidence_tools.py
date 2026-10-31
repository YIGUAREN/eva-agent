from pathlib import Path

import pytest

from eva_agent.tools.dataset import SampleDatasetInput, build_sample_dataset_tool
from eva_agent.tools.evidence import (
    EvidenceAccessError,
    ReadEvidenceInput,
    build_read_evidence_tool,
    resolve_evidence_path,
)


def test_resolve_rejects_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(EvidenceAccessError, match="escapes"):
        resolve_evidence_path(str(root), "../secret.txt")


def test_resolve_rejects_absolute_path_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(EvidenceAccessError, match="escapes"):
        resolve_evidence_path(str(root), str(outside))


def test_resolve_rejects_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()

    with pytest.raises(EvidenceAccessError, match="not a file"):
        resolve_evidence_path(str(root), "nope.txt")


async def test_read_evidence_returns_text(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "paste.json").write_text('{"query": "q"}', encoding="utf-8")
    tool = build_read_evidence_tool(default_max_bytes=1024)

    text = await tool.handler(ReadEvidenceInput(root=str(root), path="paste.json"))

    assert text == '{"query": "q"}'


async def test_read_evidence_enforces_size_limit(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "big.txt").write_text("x" * 20, encoding="utf-8")
    tool = build_read_evidence_tool(default_max_bytes=10)

    with pytest.raises(EvidenceAccessError, match="exceeds"):
        await tool.handler(ReadEvidenceInput(root=str(root), path="big.txt"))


async def test_truncate_drops_a_split_multi_byte_tail(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    # "é" is two bytes; cutting after the first byte leaves an incomplete tail.
    (root / "big.txt").write_bytes(b"a" * 20 + "é".encode() + b"tail")
    tool = build_read_evidence_tool(default_max_bytes=1024)

    text = await tool.handler(
        ReadEvidenceInput(root=str(root), path="big.txt", max_bytes=21, truncate=True, strict_truncate=True)
    )

    assert text == "a" * 20


async def test_strict_truncate_rejects_binary_before_the_tail(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "blob.bin").write_bytes(b"a" * 20 + b"\xff\xfe" + b"b" * 20)
    tool = build_read_evidence_tool(default_max_bytes=1024)

    with pytest.raises(EvidenceAccessError, match="not valid UTF-8"):
        await tool.handler(
            ReadEvidenceInput(root=str(root), path="blob.bin", max_bytes=30, truncate=True, strict_truncate=True)
        )


async def test_lenient_truncate_is_the_default(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "blob.bin").write_bytes(b"a" * 20 + b"\xff\xfe" + b"b" * 20)
    tool = build_read_evidence_tool(default_max_bytes=1024)

    text = await tool.handler(ReadEvidenceInput(root=str(root), path="blob.bin", max_bytes=30, truncate=True))

    assert text == "a" * 20 + "b" * 8


async def test_sample_dataset_limits_rows(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "dataset.csv").write_text(
        "query,response,reference\nq1,a1,r1\nq2,a2,r2\nq3,a3,r3\n",
        encoding="utf-8",
    )
    tool = build_sample_dataset_tool(max_bytes=10_000, max_rows=10)

    sample = await tool.handler(SampleDatasetInput(root=str(root), path="dataset.csv", limit=2))

    assert sample.columns == ["query", "response", "reference"]
    assert len(sample.rows) == 2
    assert sample.rows[0]["reference"] == "r1"


async def test_sample_dataset_applies_column_map(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "dataset.csv").write_text("q,a\nq1,a1\n", encoding="utf-8")
    tool = build_sample_dataset_tool(max_bytes=10_000, max_rows=10)

    sample = await tool.handler(
        SampleDatasetInput(
            root=str(root),
            path="dataset.csv",
            limit=1,
            column_map={"query": "q", "response": "a"},
        )
    )

    assert sample.rows == [{"query": "q1", "response": "a1", "reference": ""}]


async def test_sample_dataset_rejects_unknown_extension(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "dataset.txt").write_text("q,a", encoding="utf-8")
    tool = build_sample_dataset_tool(max_bytes=10_000, max_rows=10)

    with pytest.raises(ValueError, match="csv or jsonl"):
        await tool.handler(SampleDatasetInput(root=str(root), path="dataset.txt", limit=1))
