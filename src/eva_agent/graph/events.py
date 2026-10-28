"""Event sink used to persist run progress."""

from collections.abc import Awaitable, Callable
from typing import Any

EventSink = Callable[[str, str, dict[str, Any] | None], Awaitable[None]]


async def noop_event_sink(
    run_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> None:
    return None
