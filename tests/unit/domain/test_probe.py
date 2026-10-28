from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eva_agent.domain.probe import ProbeRecord, ProbeSummary


def test_probe_record_round_trip():
    record = ProbeRecord(
        index=0,
        prompt="q",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        status_code=200,
        latency_ms=12,
        response_text="a",
        input_tokens=3,
        output_tokens=2,
        usage_source="target",
    )

    restored = ProbeRecord.model_validate_json(record.model_dump_json())

    assert restored == record


def test_probe_record_rejects_negative_latency():
    with pytest.raises(ValidationError):
        ProbeRecord(index=0, prompt="q", started_at=datetime.now(UTC), latency_ms=-1)


def test_probe_summary_counts():
    summary = ProbeSummary(path="p", total=3, succeeded=2, failed=1)
    assert summary.succeeded + summary.failed == summary.total
