from eva_agent.domain.audit import AuditEvent
from eva_agent.security.audit import NoopAuditSink, sanitize_detail


def test_sanitize_redacts_secrets_truncates_strings_and_limits_collections() -> None:
    clean = sanitize_detail(
        {
            "api_key": "sk-secret",
            "nested": {"token": "abc", "note": "x" * 600},
            "items": list(range(50)),
            "flag": True,
            "none": None,
            "number": 3.5,
        }
    )
    assert clean["api_key"] == "<redacted>"
    assert clean["nested"]["token"] == "<redacted>"
    assert len(clean["nested"]["note"]) == 512
    assert clean["items"] == list(range(32))
    assert clean["flag"] is True
    assert clean["none"] is None
    assert clean["number"] == 3.5


async def test_noop_sink_accepts_any_event() -> None:
    event = AuditEvent(run_id="run-1", actor="cli:admin", action="x", outcome="allowed")
    assert await NoopAuditSink().record(event) is None
