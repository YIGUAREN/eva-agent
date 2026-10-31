import re

import eva_agent.server.schemas as schemas
from eva_agent.server.schemas import (
    redact_absolute_paths,
    redact_markdown_paths,
    sanitize_public_payload,
    sanitize_public_text,
)

LOCATOR = "data/runs/abc/aig/sess-1.json"


def test_payload_drops_a_relative_aig_locator() -> None:
    payload = {"findings": [{"evidence": [{"kind": "file", "locator": LOCATOR, "excerpt": "T04"}]}]}
    sanitized = sanitize_public_payload(payload)
    assert sanitized["findings"][0]["evidence"][0]["locator"] == "<path>"
    assert sanitized["findings"][0]["evidence"][0]["excerpt"] == "T04"


def test_payload_drops_a_bare_aig_artifact_name() -> None:
    assert sanitize_public_payload({"locator": "aig/sess-1.csv"})["locator"] == "<path>"


def test_markdown_and_text_redact_the_relative_form() -> None:
    line = f"| critical | Credential harvesting | {LOCATOR} |"
    assert redact_markdown_paths(line) == "| critical | Credential harvesting | <path> |"
    assert redact_absolute_paths(LOCATOR) == "<path>"


def test_ordinary_relative_paths_are_left_alone() -> None:
    assert sanitize_public_payload({"locator": "SKILL.md"})["locator"] == "SKILL.md"
    assert sanitize_public_payload({"locator": "docs/report.md"})["locator"] == "docs/report.md"


WINDOWS_LOCATOR = r"data\runs\abc\aig\sess-1.json"


def test_payload_drops_a_windows_separator_aig_locator() -> None:
    payload = {"findings": [{"evidence": [{"kind": "file", "locator": WINDOWS_LOCATOR, "excerpt": "T04"}]}]}
    sanitized = sanitize_public_payload(payload)
    assert sanitized["findings"][0]["evidence"][0]["locator"] == "<path>"
    assert sanitized["findings"][0]["evidence"][0]["excerpt"] == "T04"


def test_text_and_markdown_redact_the_windows_separator_form() -> None:
    assert redact_absolute_paths(WINDOWS_LOCATOR) == "<path>"
    assert redact_markdown_paths(f"| critical | {WINDOWS_LOCATOR} |") == "| critical | <path> |"


def test_uppercase_aig_segment_is_redacted() -> None:
    assert sanitize_public_payload({"locator": "AIG/sess-1.json"})["locator"] == "<path>"


def test_artifact_lookalikes_are_left_alone() -> None:
    assert sanitize_public_payload({"locator": "SKILL.md"})["locator"] == "SKILL.md"
    assert sanitize_public_payload({"locator": "docs/report.md"})["locator"] == "docs/report.md"
    assert sanitize_public_payload({"locator": "aig/sess-1.json.bak"})["locator"] == "aig/sess-1.json.bak"


MIXED_CASE_WINDOWS_LOCATOR = r"Data\Runs\abc\AIG\sess-1.csv"


def test_mixed_case_windows_locator_is_redacted_by_every_entry_point() -> None:
    assert redact_absolute_paths(MIXED_CASE_WINDOWS_LOCATOR) == "<path>"
    assert (
        redact_markdown_paths(f"| critical | {MIXED_CASE_WINDOWS_LOCATOR} |")
        == "| critical | <path> |"
    )
    assert sanitize_public_text(MIXED_CASE_WINDOWS_LOCATOR) == "<path>"
    assert sanitize_public_payload({"locator": MIXED_CASE_WINDOWS_LOCATOR})["locator"] == "<path>"


def test_markdown_redacts_a_bare_windows_absolute_path() -> None:
    locator = r"C:\Users\analyst\data\aig\sess-1.json"

    assert redact_markdown_paths(locator) == "<path>"


def test_a_sentence_ending_period_does_not_defeat_the_artifact_pattern() -> None:
    assert redact_absolute_paths("see aig/sess-1.json. Thanks") == "see <path>. Thanks"
    assert redact_absolute_paths("aig/sess-1.json.") == "<path>."
    assert redact_markdown_paths("see aig/sess-1.json. Thanks") == "see <path>. Thanks"
    assert redact_markdown_paths("aig/sess-1.json.") == "<path>."
    assert sanitize_public_text("aig/sess-1.json.") == "<path>."
    assert sanitize_public_payload({"locator": "aig/sess-1.json."})["locator"] == "<path>"


def test_public_entry_points_share_the_public_path_patterns(monkeypatch) -> None:
    sentinel = re.compile(r"SENTINEL")
    monkeypatch.setattr(schemas, "_PUBLIC_PATH_PATTERNS", (sentinel,))

    assert redact_absolute_paths("SENTINEL") == "<path>"
    assert redact_markdown_paths("SENTINEL") == "<path>"
    assert sanitize_public_text("SENTINEL") == "<path>"
    assert sanitize_public_payload("SENTINEL") == "<path>"
