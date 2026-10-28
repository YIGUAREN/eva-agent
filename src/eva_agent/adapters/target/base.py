"""Transport-independent target request and response models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from eva_agent.security.target_policy import ValidatedTarget


class TargetResponseError(ValueError):
    """Raised when a target response cannot be parsed."""


@dataclass(frozen=True)
class TargetRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


@dataclass(frozen=True)
class TargetResponse:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


@dataclass(frozen=True)
class ParsedTargetResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class TargetTransport(Protocol):
    async def send(
        self,
        request: TargetRequest,
        *,
        validated: ValidatedTarget,
        timeout_s: float,
        max_response_bytes: int,
    ) -> TargetResponse: ...
