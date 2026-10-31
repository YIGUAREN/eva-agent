from decimal import Decimal

import pytest

from eva_agent.budget.types import RunUsage
from eva_agent.domain.budget import Budget
from eva_agent.runtime.budget_decisions import (
    BudgetOverrunRequired,
    InvalidBudgetExtension,
    ensure_summarize_allowed,
    validate_extension,
)


def test_summarize_requires_confirmation_only_when_the_tail_overruns() -> None:
    ensure_summarize_allowed(
        finalize_estimate=Decimal("0.2"),
        remaining=Decimal("0.3"),
        confirm_overrun=False,
    )
    with pytest.raises(BudgetOverrunRequired) as exc:
        ensure_summarize_allowed(
            finalize_estimate=Decimal("0.2"),
            remaining=Decimal("0.1"),
            confirm_overrun=False,
        )
    assert exc.value.finalize_estimate == Decimal("0.2")
    assert exc.value.remaining == Decimal("0.1")
    ensure_summarize_allowed(
        finalize_estimate=Decimal("0.2"),
        remaining=Decimal("0.1"),
        confirm_overrun=True,
    )


def test_extend_rejects_each_limit_that_is_not_strictly_larger() -> None:
    usage = RunUsage(
        spent_usd=Decimal("0.5"),
        spent_wall_clock_s=10.0,
        target_requests=3,
        aig_requests=4,
    )
    with pytest.raises(InvalidBudgetExtension) as exc:
        validate_extension(
            Budget(max_cost_usd=0.5, max_wall_clock_s=20.0, max_target_requests=10),
            usage,
        )
    assert exc.value.field == "max_cost_usd"
    with pytest.raises(InvalidBudgetExtension) as exc:
        validate_extension(
            Budget(max_cost_usd=1.0, max_wall_clock_s=10.0, max_target_requests=10),
            usage,
        )
    assert exc.value.field == "max_wall_clock_s"
    with pytest.raises(InvalidBudgetExtension) as exc:
        validate_extension(
            Budget(max_cost_usd=1.0, max_wall_clock_s=20.0, max_target_requests=4),
            usage,
        )
    assert exc.value.field == "max_target_requests"
    validate_extension(
        Budget(max_cost_usd=1.0, max_wall_clock_s=20.0, max_target_requests=5),
        usage,
    )


def test_extend_rejection_messages_do_not_embed_spend_amounts() -> None:
    usage = RunUsage(
        spent_usd=Decimal("0.5"),
        spent_wall_clock_s=10.0,
        target_requests=3,
        aig_requests=4,
    )
    budgets = [
        Budget(max_cost_usd=0.5, max_wall_clock_s=20.0, max_target_requests=10),
        Budget(max_cost_usd=1.0, max_wall_clock_s=10.0, max_target_requests=10),
        Budget(max_cost_usd=1.0, max_wall_clock_s=20.0, max_target_requests=4),
    ]
    for budget in budgets:
        with pytest.raises(InvalidBudgetExtension) as exc:
            validate_extension(budget, usage)
        assert not any(character.isdigit() for character in str(exc.value))


def test_extend_reports_bad_fields_in_cost_wall_requests_order() -> None:
    usage = RunUsage(
        spent_usd=Decimal("0.5"),
        spent_wall_clock_s=10.0,
        target_requests=3,
        aig_requests=4,
    )

    def rejected_field(budget: Budget) -> str:
        with pytest.raises(InvalidBudgetExtension) as exc:
            validate_extension(budget, usage)
        return exc.value.field

    # All three limits are too small: cost is reported first.
    assert (
        rejected_field(Budget(max_cost_usd=0.5, max_wall_clock_s=10.0, max_target_requests=4))
        == "max_cost_usd"
    )
    assert (
        rejected_field(Budget(max_cost_usd=1.0, max_wall_clock_s=10.0, max_target_requests=4))
        == "max_wall_clock_s"
    )
    assert (
        rejected_field(Budget(max_cost_usd=1.0, max_wall_clock_s=20.0, max_target_requests=4))
        == "max_target_requests"
    )
