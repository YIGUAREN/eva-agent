"""Transport-independent target request and response models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from eva_agent.domain.target import TargetSpec
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


class TargetAdapter(Protocol):
    def build_request(
        self,
        target: TargetSpec,
        prompt: str,
        *,
        headers: dict[str, str],
    ) -> TargetRequest: ...

    def parse_response(self, target: TargetSpec, response: TargetResponse) -> ParsedTargetResponse: ...


def load_json_body(response: TargetResponse) -> Any:
    try:
        return json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetResponseError("response body is not valid JSON") from exc


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
