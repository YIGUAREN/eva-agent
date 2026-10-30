from pathlib import Path

from eva_agent.domain.audit import AuditEvent
from eva_agent.domain.budget import Budget
from eva_agent.domain.run import InputKind, PastedInput, RunInput
from eva_agent.storage.audit import AuditRepository
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.repository import RunRepository


async def test_record_and_list_for_run(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'eva.db').as_posix()}")
    await init_db(engine)
    session_factory = create_session_factory(engine)
    await RunRepository(session_factory).create(
        "run-1",
        RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q", response="a")),
        Budget(max_cost_usd=1, max_wall_clock_s=10, max_target_requests=1),
    )
    repo = AuditRepository(session_factory)
    await repo.record(
        AuditEvent(
            run_id="run-1",
            actor="cli:admin",
            action="active_scan.requested",
            outcome="allowed",
            target="demo",
            authorization_ref="demo",
            scan_kind="model_redteam_report",
            detail={"api_key": "sk-secret", "note": "x" * 600},
        )
    )
    await repo.record(
        AuditEvent(run_id="run-1", actor="cli:admin", action="active_scan.completed", outcome="completed")
    )

    rows = await repo.list_for_run("run-1")
    assert [row.action for row in rows] == ["active_scan.requested", "active_scan.completed"]
    assert rows[0].detail_json["api_key"] == "<redacted>"
    assert len(rows[0].detail_json["note"]) == 512
    await engine.dispose()
