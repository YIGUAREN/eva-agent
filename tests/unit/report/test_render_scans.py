from eva_agent.report.render import _security_lines


def render(active: dict) -> str:
    return "\n".join(_security_lines({"passive": {"by_grader": {}}, "active": active}))


def test_agent_scan_metrics_show_the_kind_and_risk_type() -> None:
    text = render({"kind": "agent_scan", "status": "completed", "risk_type": "high", "jailbreak": 1, "total": 12})
    assert "扫描种类：`agent_scan`" in text
    assert "风险类型：`high`" in text


def test_model_redteam_metrics_default_to_the_old_kind_label() -> None:
    text = render({"status": "completed", "jailbreak": 0, "total": 4})
    assert "扫描种类：`model_redteam_report`" in text
    assert "风险类型" not in text
