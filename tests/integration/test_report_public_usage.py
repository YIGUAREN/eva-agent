from eva_agent.server.routes.reports import public_report_usage
from eva_agent.server.schemas import sanitize_public_payload


def test_public_payload_keeps_only_the_four_usage_keys() -> None:
    payload = {
        "usage": {
            "estimated": True,
            "cached": 2,
            "fresh": 1,
            "budget_enforced": True,
            "spent_usd": 0.9,
        },
        "active_scan": {"kind": "skill_scan", "raw_result_path": "/srv/aig/x.json"},
    }
    assert public_report_usage(payload) == {
        "estimated": True,
        "cached": 2,
        "fresh": 1,
        "budget_enforced": True,
    }
    public = sanitize_public_payload(payload)
    assert "raw_result_path" not in public["active_scan"]
