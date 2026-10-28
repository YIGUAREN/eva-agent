"""Tools that call a target and read back probe records."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from eva_agent.adapters.target import build_adapter
from eva_agent.adapters.target.base import TargetAdapter, TargetResponseError, TargetTransport
from eva_agent.adapters.target.http import TargetResponseTooLargeError
from eva_agent.adapters.target.secrets import MissingSecretError, resolve_headers
from eva_agent.adapters.target.tokens import TokenEstimationError, estimate_tokens
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.target import TargetSpec
from eva_agent.runtime.probe import ProbeStore
from eva_agent.security.target_policy import SecurityPolicyError, ValidatedTarget
from eva_agent.tools.evidence import resolve_evidence_path
from eva_agent.tools.registry import ToolSpec

AdapterFactory = Callable[[TargetSpec], TargetAdapter]
UsageSource = Literal["target", "estimated", "missing"]
ERROR_EXCERPT_CHARS = 512


class TargetValidator(Protocol):
    async def validate(self, target: TargetSpec) -> ValidatedTarget: ...


class CallTargetInput(BaseModel):
    index: int = Field(ge=0)
    target: TargetSpec
    prompt: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    timeout_s: float = Field(gt=0)
    max_response_bytes: int | None = Field(default=None, gt=0)


class ReadTargetProbeInput(BaseModel):
    root: str = Field(min_length=1)
    run_id: str | None = None


class ReadTargetProbeOutput(BaseModel):
    records: list[ProbeRecord]


def build_call_target_tool(
    *,
    policy: TargetValidator,
    transport: TargetTransport,
    env: Mapping[str, str],
    adapter_factory: AdapterFactory = build_adapter,
    default_max_response_bytes: int = 1_048_576,
    token_encoding: str = "cl100k_base",
) -> ToolSpec:
    async def handler(payload: CallTargetInput) -> ProbeRecord:
        return await _call_target(
            payload,
            policy=policy,
            transport=transport,
            env=env,
            adapter_factory=adapter_factory,
            default_max_response_bytes=default_max_response_bytes,
            token_encoding=token_encoding,
        )

    return ToolSpec(
        name="call_target",
        description="Send one target request and return a probe record.",
        input_model=CallTargetInput,
        output_model=ProbeRecord,
        permission="admin",
        timeout_s=600.0,
        side_effects=True,
        idempotent=False,
        handler=handler,
    )


async def _call_target(
    payload: CallTargetInput,
    *,
    policy: TargetValidator,
    transport: TargetTransport,
    env: Mapping[str, str],
    adapter_factory: AdapterFactory,
    default_max_response_bytes: int,
    token_encoding: str,
) -> ProbeRecord:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    status_code: int | None = None
    response_text = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage_source: UsageSource = "missing"
    error: str | None = None
    latency_ms = 0

    try:
        validated = await policy.validate(payload.target)
        adapter = adapter_factory(payload.target)
        headers = resolve_headers(payload.target, env)
        request = adapter.build_request(payload.target, payload.prompt, headers=headers)
        response = await transport.send(
            request,
            validated=validated,
            timeout_s=payload.timeout_s,
            max_response_bytes=payload.max_response_bytes or default_max_response_bytes,
        )
        status_code = response.status_code
        latency_ms = _elapsed_ms(started)

        if 300 <= status_code < 400:
            response_text = _decode_excerpt(response.body)
            error = "redirect_not_allowed"
        elif not 200 <= status_code < 300:
            response_text = _decode_excerpt(response.body)
            error = f"http {status_code}"
        else:
            parsed = adapter.parse_response(payload.target, response)
            response_text = parsed.text
            input_tokens, output_tokens, usage_source = _resolve_tokens(
                parsed.input_tokens, parsed.output_tokens, parsed.text, payload.prompt, token_encoding
            )
    except Exception as exc:
        latency_ms = _elapsed_ms(started)
        error = _error_summary(exc)

    return ProbeRecord(
        index=payload.index,
        prompt=payload.prompt,
        started_at=started_at,
        status_code=status_code,
        latency_ms=latency_ms,
        response_text=response_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_source=usage_source,
        error=error,
    )


def build_read_target_probe_tool() -> ToolSpec:
    async def handler(payload: ReadTargetProbeInput) -> ReadTargetProbeOutput:
        path = resolve_evidence_path(payload.root, "target_probe.jsonl")
        return ReadTargetProbeOutput(records=ProbeStore(path).read())

    return ToolSpec(
        name="read_target_probe",
        description="Read target_probe.jsonl for one run.",
        input_model=ReadTargetProbeInput,
        output_model=ReadTargetProbeOutput,
        permission="internal",
        timeout_s=10.0,
        side_effects=False,
        idempotent=True,
        handler=handler,
    )


def _resolve_tokens(
    input_tokens: int | None,
    output_tokens: int | None,
    text: str,
    prompt: str,
    encoding: str,
) -> tuple[int | None, int | None, UsageSource]:
    if input_tokens is not None and output_tokens is not None:
        return input_tokens, output_tokens, "target"
    if not text:
        return None, None, "missing"
    try:
        return estimate_tokens(prompt, encoding), estimate_tokens(text, encoding), "estimated"
    except TokenEstimationError:
        return None, None, "missing"


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _decode_excerpt(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")[:ERROR_EXCERPT_CHARS]


def _error_summary(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, TargetResponseTooLargeError):
        return "response_too_large"
    if isinstance(exc, SecurityPolicyError):
        return f"policy: {exc}"
    if isinstance(exc, MissingSecretError):
        return f"secret: {exc}"
    if isinstance(exc, TargetResponseError):
        return f"response: {exc}"
    return f"{type(exc).__name__}: {exc}"
