import pytest
from pydantic import ValidationError

from eva_agent.domain.budget import Budget


def make_budget(**overrides: object) -> Budget:
    data: dict[str, object] = {
        "max_cost_usd": 1.0,
        "max_wall_clock_s": 600.0,
        "max_target_requests": 10,
    }
    data.update(overrides)
    return Budget(**data)  # type: ignore[arg-type]


def test_valid_budget_round_trips_json() -> None:
    budget = make_budget()

    assert Budget.model_validate(budget.model_dump(mode="json")) == budget


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_cost_usd", 0),
        ("max_cost_usd", -1.0),
        ("max_wall_clock_s", 0),
        ("max_wall_clock_s", -5.0),
        ("max_target_requests", 0),
        ("max_target_requests", -1),
    ],
)
def test_rejects_non_positive_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_budget(**{field: value})


def test_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        make_budget(unexpected=1)


def test_budget_is_immutable() -> None:
    budget = make_budget()

    with pytest.raises(ValidationError):
        budget.max_cost_usd = 2.0  # type: ignore[misc]
