"""Minimal typed tool registry for phase 1."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel] | None
    permission: Literal["internal", "admin", "public_read"]
    timeout_s: float
    side_effects: bool
    idempotent: bool
    handler: Callable[[Any], Awaitable[Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool '{name}'") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def call(self, name: str, payload: BaseModel) -> Any:
        spec = self.get(name)
        if not isinstance(payload, spec.input_model):
            raise TypeError(f"tool '{name}' expects {spec.input_model.__name__}, got {type(payload).__name__}")

        started = time.perf_counter()
        run_id = getattr(payload, "run_id", None)
        try:
            result = await asyncio.wait_for(spec.handler(payload), timeout=spec.timeout_s)
        except TimeoutError as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "tool_call",
                extra={"tool": name, "run_id": run_id, "duration_ms": elapsed_ms, "status": "timeout"},
            )
            raise TimeoutError(f"tool '{name}' timed out after {spec.timeout_s}s") from exc
        except Exception:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "tool_call",
                extra={"tool": name, "run_id": run_id, "duration_ms": elapsed_ms, "status": "error"},
            )
            raise

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "tool_call",
            extra={"tool": name, "run_id": run_id, "duration_ms": elapsed_ms, "status": "ok"},
        )
        return result
