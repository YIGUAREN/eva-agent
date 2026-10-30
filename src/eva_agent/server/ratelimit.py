"""In-memory login throttling for the single administrator."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable


class LoginRateLimiter:
    def __init__(
        self,
        *,
        max_failures: int = 5,
        window_s: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_failures = max_failures
        self._window_s = window_s
        self._clock = clock
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str) -> None:
        cutoff = self._clock() - self._window_s
        failures = self._failures[key]
        while failures and failures[0] <= cutoff:
            failures.popleft()

    def is_blocked(self, key: str) -> bool:
        self._prune(key)
        return len(self._failures[key]) >= self._max_failures

    def record_failure(self, key: str) -> None:
        self._prune(key)
        self._failures[key].append(self._clock())

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)
