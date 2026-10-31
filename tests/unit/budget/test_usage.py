from types import SimpleNamespace

from eva_agent.budget.usage import record_usage, track_usage


def test_track_usage_sums_prompt_and_completion_tokens() -> None:
    with track_usage() as totals:
        record_usage(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7)))
        record_usage(SimpleNamespace(usage={"prompt_tokens": 3, "completion_tokens": 2}))
    assert totals.input_tokens == 14
    assert totals.output_tokens == 9


def test_record_usage_outside_a_tracking_scope_is_a_noop() -> None:
    record_usage(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=100)))
    with track_usage() as totals:
        pass
    assert totals.input_tokens == 0
