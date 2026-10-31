"""Collect provider-reported token usage inside one graph invocation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0

    def add_response(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        if isinstance(usage, Mapping):
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
        else:
            prompt = getattr(usage, "prompt_tokens", 0)
            completion = getattr(usage, "completion_tokens", 0)
        self.input_tokens += int(prompt or 0)
        self.output_tokens += int(completion or 0)


_CURRENT: ContextVar[UsageTotals | None] = ContextVar("eva_usage_totals", default=None)


@contextmanager
def track_usage() -> Iterator[UsageTotals]:
    totals = UsageTotals()
    token = _CURRENT.set(totals)
    try:
        yield totals
    finally:
        _CURRENT.reset(token)


def record_usage(response: Any) -> None:
    totals = _CURRENT.get()
    if totals is not None:
        totals.add_response(response)
