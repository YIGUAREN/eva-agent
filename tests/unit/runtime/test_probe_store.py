from datetime import UTC, datetime
from pathlib import Path

import pytest

from eva_agent.domain.probe import ProbeRecord
from eva_agent.runtime.probe import ProbeStore, ProbeStoreError


def record(index: int, *, status: int | None = 200, error: str | None = None) -> ProbeRecord:
    return ProbeRecord(
        index=index,
        prompt="q",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        status_code=status,
        latency_ms=10,
        response_text="a",
        error=error,
    )


def test_append_read_and_summary(tmp_path: Path):
    store = ProbeStore(tmp_path / "target_probe.jsonl")
    store.reset()
    store.append(record(0))
    store.append(record(1, status=500, error="http 500"))
    store.append(record(2, status=None, error="timeout"))

    records = store.read()
    summary = store.summary()

    assert [item.index for item in records] == [0, 1, 2]
    assert summary.total == 3
    assert summary.succeeded == 1
    assert summary.failed == 2
    assert summary.path.endswith("target_probe.jsonl")


def test_reset_clears_existing_records(tmp_path: Path):
    store = ProbeStore(tmp_path / "target_probe.jsonl")
    store.reset()
    store.append(record(0))

    store.reset()

    assert store.read() == []


def test_missing_file_reads_empty(tmp_path: Path):
    assert ProbeStore(tmp_path / "missing.jsonl").read() == []


def test_malformed_line_rejected(tmp_path: Path):
    path = tmp_path / "target_probe.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ProbeStoreError, match="invalid probe record"):
        ProbeStore(path).read()
