import pytest

from eva_agent.adapters.aig.errors import AigResultError
from eva_agent.adapters.aig.models import parse_agent_result, parse_skill_result


def skill_payload(**result_overrides) -> dict:
    result = {
        "readme": "report",
        "score": 100,
        "language": "YAML",
        "start_time": 1793422905.46,
        "end_time": 1793422956.87,
        "results": [],
        "llm": "deepseek-chat",
    }
    result.update(result_overrides)
    return {"id": "msg-1", "type": "resultUpdate", "timestamp": 1793422956, "result": result}


def agent_payload(**result_overrides) -> dict:
    result = {
        "schema_version": "agent-security-report@1",
        "agent_name": "http",
        "agent_type": "http",
        "model_name": "",
        "start_time": 0,
        "end_time": 0,
        "plugins": [],
        "score": 100,
        "risk_type": "safe",
        "total_tests": 12,
        "vulnerable_tests": 0,
        "results": [],
        "owasp_agentic_2026_top10": [],
        "report_description": "collected",
        "language": "en",
    }
    result.update(result_overrides)
    return {"id": "msg-2", "type": "resultUpdate", "timestamp": 1, "result": result}


def test_skill_result_parses_the_direct_result_object() -> None:
    report = parse_skill_result(skill_payload())
    assert report.score == 100
    assert report.language == "YAML"
    assert report.llm == "deepseek-chat"
    assert report.results == []


def test_skill_result_keeps_a_zero_score_with_a_finding() -> None:
    payload = skill_payload(
        score=0,
        results=[
            {
                "title": "Exfiltration",
                "description": "Sends secrets out.",
                "risk_type": "T04: Embedded Malicious Code",
                "level": "Critical",
                "suggestion": "Remove it.",
                "file": "ssh-collector/SKILL.md",
                "line_start": 13,
                "line_end": 21,
            }
        ],
    )
    report = parse_skill_result(payload)
    finding = report.results[0]
    assert report.score == 0
    assert finding.level == "Critical"
    assert finding.file == "ssh-collector/SKILL.md"


def test_skill_finding_must_have_a_description() -> None:
    payload = skill_payload(results=[{"title": "t", "description": "", "risk_type": "T04"}])
    with pytest.raises(AigResultError):
        parse_skill_result(payload)


def test_skill_finding_file_must_stay_inside_the_package() -> None:
    for bad in ("/etc/passwd", "../escape.md", "a/../../b.md"):
        payload = skill_payload(results=[{"title": "t", "description": "d", "risk_type": "T04", "file": bad}])
        with pytest.raises(AigResultError):
            parse_skill_result(payload)


def test_result_must_be_an_object() -> None:
    with pytest.raises(AigResultError, match="no structured result object"):
        parse_skill_result({"id": "m", "result": []})


def test_agent_result_parses_the_report_fields() -> None:
    report = parse_agent_result(
        agent_payload(
            score=55,
            risk_type="high",
            vulnerable_tests=1,
            results=[
                {
                    "id": "f-001",
                    "type": "tool_abuse",
                    "title": "Tool abuse",
                    "description": "Calls a shell tool.",
                    "level": "HIGH",
                    "owasp": ["ASI02"],
                    "suggestion": "Restrict tools.",
                    "conversation": [{"prompt": "p", "response": "r"}],
                }
            ],
        )
    )
    assert report.score == 55
    assert report.risk_type == "high"
    assert report.total_tests == 12
    assert report.vulnerable_tests == 1
    assert report.results[0].level == "HIGH"


@pytest.mark.parametrize("level", ["high", "critical", "", "MEDIUM "])
def test_agent_finding_level_must_be_canonical(level: str) -> None:
    payload = agent_payload(
        results=[{"title": "t", "description": "d", "level": level, "risk_type": "x"}]
    )
    with pytest.raises(AigResultError):
        parse_agent_result(payload)


@pytest.mark.parametrize("score", [-1, 101])
def test_score_must_be_in_range(score: int) -> None:
    with pytest.raises(AigResultError):
        parse_agent_result(agent_payload(score=score))
