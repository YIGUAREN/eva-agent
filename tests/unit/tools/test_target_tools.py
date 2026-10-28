import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import h11
import pytest

from eva_agent.adapters.target import tokens as tokens_module
from eva_agent.adapters.target.base import TargetRequest, TargetResponse
from eva_agent.domain.probe import ProbeRecord
from eva_agent.domain.target import TargetKind, TargetSpec
from eva_agent.runtime.probe import ProbeStore
from eva_agent.security.target_policy import SecurityPolicyError, ValidatedTarget
from eva_agent.tools.target import (
    CallTargetInput,
    ReadTargetProbeInput,
    build_call_target_tool,
    build_read_target_probe_tool,
)


class FixedEncoder:
    """Offline stand-in for a tiktoken encoding."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


@pytest.fixture(autouse=True)
def offline_tiktoken(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # tiktoken downloads vocabularies on a cache miss; tests must never reach the network.
    tokens_module._encoding.cache_clear()
    monkeypatch.setattr(tokens_module.tiktoken, "get_encoding", lambda name: FixedEncoder())
    yield
    tokens_module._encoding.cache_clear()


def openai_target(api_key_env: str | None = None) -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1",
        api_key_env=api_key_env,
    )


def validated() -> ValidatedTarget:
    return ValidatedTarget(
        url="http://127.0.0.1:8100/v1",
        scheme="http",
        host="127.0.0.1",
        port=8100,
        addresses=("127.0.0.1",),
        demo=True,
    )


class FakePolicy:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def validate(self, target: TargetSpec) -> ValidatedTarget:
        if self.error is not None:
            raise self.error
        return validated()


class FakeTransport:
    def __init__(self, response: TargetResponse | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[TargetRequest, ValidatedTarget, float, int]] = []

    async def send(
        self,
        request: TargetRequest,
        *,
        validated: ValidatedTarget,
        timeout_s: float,
        max_response_bytes: int,
    ) -> TargetResponse:
        self.calls.append((request, validated, timeout_s, max_response_bytes))
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def response_body(content: str, *, usage: bool = True) -> bytes:
    payload: dict[str, Any] = {"choices": [{"message": {"content": content}}]}
    if usage:
        payload["usage"] = {"prompt_tokens": 3, "completion_tokens": 2}
    return json.dumps(payload).encode()


def build(policy: FakePolicy, transport: FakeTransport, env: dict[str, str] | None = None):
    return build_call_target_tool(
        policy=cast(Any, policy),
        transport=transport,
        env=env or {},
    )


async def test_success_records_target_usage():
    transport = FakeTransport(TargetResponse(200, {}, response_body("ok")))
    tool = build(FakePolicy(), transport)

    record = await tool.handler(
        CallTargetInput(index=0, target=openai_target(), prompt="q", run_id="r", timeout_s=5)
    )

    assert record.status_code == 200
    assert record.response_text == "ok"
    assert record.input_tokens == 3
    assert record.output_tokens == 2
    assert record.usage_source == "target"
    assert record.error is None
    assert transport.calls[0][0].url == "http://127.0.0.1:8100/v1/chat/completions"
    assert transport.calls[0][2] == 5


async def test_missing_usage_is_estimated():
    transport = FakeTransport(TargetResponse(200, {}, response_body("ok", usage=False)))
    tool = build(FakePolicy(), transport)

    record = await tool.handler(
        CallTargetInput(index=0, target=openai_target(), prompt="question", run_id="r", timeout_s=5)
    )

    assert record.usage_source == "estimated"
    assert record.input_tokens is not None and record.input_tokens > 0
    assert record.output_tokens is not None and record.output_tokens > 0


async def test_empty_response_marks_usage_missing():
    transport = FakeTransport(TargetResponse(200, {}, response_body("", usage=False)))
    tool = build(FakePolicy(), transport)

    record = await tool.handler(
        CallTargetInput(index=0, target=openai_target(), prompt="q", run_id="r", timeout_s=5)
    )

    assert record.usage_source == "missing"
    assert record.input_tokens is None


async def test_http_error_is_recorded():
    transport = FakeTransport(TargetResponse(500, {}, b"boom"))
    tool = build(FakePolicy(), transport)

    record = await tool.handler(
        CallTargetInput(index=1, target=openai_target(), prompt="q", run_id="r", timeout_s=5)
    )

    assert record.status_code == 500
    assert record.error == "http 500"
    assert record.response_text == "boom"
    assert record.input_tokens is None


async def test_redirect_is_not_followed():
    transport = FakeTransport(TargetResponse(302, {"location": "http://evil"}, b""))
    tool = build(FakePolicy(), transport)

    record = await tool.handler(
        CallTargetInput(index=0, target=openai_target(), prompt="q", run_id="r", timeout_s=5)
    )

    assert record.error == "redirect_not_allowed"


async def test_policy_error_is_recorded():
    tool = build(FakePolicy(SecurityPolicyError("denied")), FakeTransport())

    record = await tool.handler(
        CallTargetInput(index=0, target=openai_target(), prompt="q", run_id="r", timeout_s=5)
    )

    assert record.status_code is None
    assert record.error == "policy: denied"
    assert record.latency_ms >= 0


async def test_missing_secret_is_recorded():
    transport = FakeTransport(TargetResponse(200, {}, response_body("ok")))
    tool = build(FakePolicy(), transport)

    record = await tool.handler(
        CallTargetInput(index=0, target=openai_target("API_KEY"), prompt="q", run_id="r", timeout_s=5)
    )

    assert record.error == "secret: environment variable API_KEY is not set"
    assert transport.calls == []


async def test_timeout_is_recorded():
    tool = build(FakePolicy(), FakeTransport(error=TimeoutError()))

    record = await tool.handler(
        CallTargetInput(index=0, target=openai_target(), prompt="q", run_id="r", timeout_s=5)
    )

    assert record.error == "timeout"


async def test_unclassified_error_does_not_leak_secret():
    secret = "sk-live-do-not-log"
    # httpcore re-raises h11 errors verbatim, including the Authorization header value.
    leak = h11.LocalProtocolError(f"Illegal header value b'Bearer {secret}\\n'")
    tool = build(FakePolicy(), FakeTransport(error=leak), env={"API_KEY": secret})

    record = await tool.handler(
        CallTargetInput(
            index=0, target=openai_target("API_KEY"), prompt="q", run_id="r", timeout_s=5
        )
    )

    assert record.status_code is None
    assert record.error == "LocalProtocolError"
    assert secret not in str(record)
    assert secret not in record.model_dump_json()
    assert all(secret not in str(value) for value in record.model_dump().values())


async def test_render_error_is_recorded():
    tool = build(FakePolicy(), FakeTransport(TargetResponse(200, {}, b"not json")))

    record = await tool.handler(
        CallTargetInput(index=0, target=openai_target(), prompt="q", run_id="r", timeout_s=5)
    )

    assert record.status_code == 200
    assert record.error is not None and record.error.startswith("response:")


def sample_record(index: int) -> ProbeRecord:
    return ProbeRecord(
        index=index,
        prompt="q",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        status_code=200,
        latency_ms=10,
    )


async def test_read_target_probe_returns_records(tmp_path: Path):
    root = tmp_path / "runs" / "r"
    store = ProbeStore(root / "target_probe.jsonl")
    store.reset()
    store.append(sample_record(0))
    store.append(sample_record(1))

    tool = build_read_target_probe_tool()
    output = await tool.handler(ReadTargetProbeInput(root=str(root)))

    assert [record.index for record in output.records] == [0, 1]
