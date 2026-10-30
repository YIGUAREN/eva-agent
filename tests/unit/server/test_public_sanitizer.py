"""Unit tests for public sanitization of free text, payloads, and objectives."""

from datetime import UTC, datetime

from eva_agent.server.schemas import (
    public_run_detail,
    redact_internal_addresses_and_secrets,
    sanitize_public_payload,
    sanitize_public_text,
)
from eva_agent.storage.models import RunRow


def _row(*, objective: str) -> RunRow:
    return RunRow(
        id="run_test",
        status="completed",
        input_json={"objective": objective},
        budget_json={},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_spaced_windows_path_does_not_leak_its_tail() -> None:
    text = "report at C:\\Users\\John Doe\\ds.jsonl#3 now"

    assert sanitize_public_text(text) == "report at <path>#3 now"


def test_spaced_posix_path_does_not_leak_its_tail() -> None:
    text = "see /Users/John Doe/ds.jsonl#3 for details"

    assert sanitize_public_text(text) == "see <path>#3 for details"


def test_spaced_unc_path_is_replaced() -> None:
    assert sanitize_public_text("\\\\server\\share\\My File\\a.jsonl") == "<path>"


def test_windows_path_in_prose_terminates_at_the_word_boundary() -> None:
    assert sanitize_public_text("run at D:\\data now") == "run at <path> now"


def test_urls_fractions_and_dates_survive() -> None:
    text = "2/3 at 2026-01-01 https://example.com/a"

    assert sanitize_public_text(text) == text


def test_loopback_and_private_addresses_are_redacted() -> None:
    for value, raw in (
        ("http://10.0.0.5:8088/x", "10.0.0.5"),
        ("127.0.0.1:9000", "127.0.0.1"),
        ("192.168.1.4", "192.168.1.4"),
        ("172.16.5.5:443", "172.16.5.5"),
        ("http://localhost:8080/y", "localhost"),
        ("http://[::1]:8000/z", "::1"),
    ):
        redacted = sanitize_public_text(value)
        assert "<internal>" in redacted, value
        assert raw not in redacted, value


def test_extra_internal_host_is_redacted_as_a_whole_token() -> None:
    redacted = redact_internal_addresses_and_secrets(
        "call http://aig.internal:8100/v1 now", internal_hosts=("aig.internal",)
    )

    assert "<internal>" in redacted
    assert "aig.internal" not in redacted


def test_secrets_are_redacted() -> None:
    text = "key sk-secret12345 header Bearer abcdefgh123 api_key=supersecret password: hunter2"

    redacted = sanitize_public_text(text)

    assert "sk-secret12345" not in redacted
    assert "abcdefgh123" not in redacted
    assert "supersecret" not in redacted
    assert "hunter2" not in redacted
    assert redacted.count("<secret>") == 4


def test_payload_redacts_addresses_and_secrets_but_keeps_harmless_text() -> None:
    payload = {
        "url": "http://127.0.0.1:8088/agent",
        "note": "token=abcdef123",
        "keep": "value",
    }

    cleaned = sanitize_public_payload(payload)

    assert isinstance(cleaned, dict)
    assert "127.0.0.1" not in cleaned["url"]
    assert "<internal>" in cleaned["url"]
    assert cleaned["note"] == "token=<secret>"
    assert cleaned["keep"] == "value"


def test_public_run_detail_sanitizes_the_objective() -> None:
    objective = (
        "inspect C:\\Users\\John Doe\\ds.jsonl at http://10.0.0.5:8088/x with sk-abcdefgh123"
    )

    detail = public_run_detail(_row(objective=objective))

    cleaned = detail["objective"]
    assert "<path>" in cleaned
    assert "<internal>" in cleaned
    assert "<secret>" in cleaned
    for raw in ("John Doe", "ds.jsonl", "10.0.0.5", "sk-abcdefgh123"):
        assert raw not in cleaned


def test_public_run_detail_honors_configured_internal_hosts() -> None:
    detail = public_run_detail(
        _row(objective="scan http://aig.internal:8100/v1 now"),
        internal_hosts=("aig.internal",),
    )

    assert "aig.internal" not in detail["objective"]
    assert "<internal>" in detail["objective"]
