"""Contract tests: the committed fixtures must parse with the shipped parsers."""

import json
from pathlib import Path

from eva_agent.adapters.aig.models import parse_skill_result

FIXTURES = Path(__file__).parent / "fixtures" / "aig"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_benign_skill_fixture_parses() -> None:
    report = parse_skill_result(load("skill_scan_result.json"))
    assert report.score == 100
    assert report.results == []
    assert report.language == "YAML"


def test_malicious_skill_fixture_parses() -> None:
    report = parse_skill_result(load("skill_scan_finding_result.json"))
    assert report.score == 0
    assert len(report.results) == 1
    finding = report.results[0]
    assert finding.risk_type == "T04: Embedded Malicious Code"
    assert finding.level == "Critical"
    assert finding.file == "ssh-collector/SKILL.md"
    assert (finding.line_start, finding.line_end) == (13, 21)
