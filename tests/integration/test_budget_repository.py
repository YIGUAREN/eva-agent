from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError

from eva_agent.budget.types import RunUsage
from eva_agent.storage.budget import BudgetDecisionRepository, RunUsageRepository
from eva_agent.storage.db import create_engine, create_session_factory, init_db
from eva_agent.storage.repository import RunRepository
from tests.support.runtime import budget, fake_config, paste_input


async def test_usage_round_trip_and_decision_idempotency(tmp_path) -> None:
    config = fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    factory = create_session_factory(engine)
    await RunRepository(factory).create("run-1", paste_input(), budget())

    usage = RunUsage(spent_usd=Decimal("0.123456"), reserved_usd=Decimal("0.000001"), target_requests=2)
    await RunUsageRepository(factory).save("run-1", usage)
    loaded = await RunUsageRepository(factory).get("run-1")
    assert loaded is not None
    assert loaded.spent_usd == Decimal("0.123456")

    decisions = BudgetDecisionRepository(factory)
    assert await decisions.record("run-1", 1, "gate", {"finalize_estimate": 0.1}) is True
    assert await decisions.record("run-1", 1, "gate", {"finalize_estimate": 0.1}) is False
    assert await decisions.next_seq("run-1") == 2
    gate = await decisions.latest_gate("run-1")
    assert gate is not None and gate.seq == 1
    await engine.dispose()


async def test_record_reraises_a_conflicting_payload(tmp_path) -> None:
    config = fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    factory = create_session_factory(engine)
    await RunRepository(factory).create("run-2", paste_input(), budget())
    decisions = BudgetDecisionRepository(factory)

    assert await decisions.record("run-2", 1, "gate", {"finalize_estimate": 0.1}) is True
    with pytest.raises(IntegrityError):
        await decisions.record("run-2", 1, "summarize", {"confirm_overrun": True})
    stored = await decisions.get("run-2", 1)
    assert stored is not None and stored.action == "gate"
    await engine.dispose()


async def test_record_reraises_other_integrity_errors(tmp_path) -> None:
    config = fake_config(tmp_path)
    engine = create_engine(config.storage.url)
    await init_db(engine)
    factory = create_session_factory(engine)
    decisions = BudgetDecisionRepository(factory)

    # No matching row exists, so the foreign-key violation must not be reported as a duplicate.
    with pytest.raises(IntegrityError):
        await decisions.record("missing-run", 1, "gate", {"finalize_estimate": 0.1})
    await engine.dispose()
