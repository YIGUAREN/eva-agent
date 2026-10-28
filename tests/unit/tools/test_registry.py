import asyncio

import pytest
from pydantic import BaseModel

from eva_agent.tools.registry import ToolRegistry, ToolSpec


class AddInput(BaseModel):
    a: int
    b: int
    run_id: str | None = None


class AddOutput(BaseModel):
    value: int


async def add(payload: AddInput) -> AddOutput:
    return AddOutput(value=payload.a + payload.b)


async def slow(payload: AddInput) -> AddOutput:
    await asyncio.sleep(0.05)
    return AddOutput(value=0)


def make_spec(name: str = "add", *, timeout_s: float = 1.0, handler=add) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="add two integers",
        input_model=AddInput,
        output_model=AddOutput,
        permission="internal",
        timeout_s=timeout_s,
        side_effects=False,
        idempotent=True,
        handler=handler,
    )


async def test_register_and_call() -> None:
    registry = ToolRegistry()
    registry.register(make_spec())

    result = await registry.call("add", AddInput(a=2, b=3))

    assert result == AddOutput(value=5)
    assert registry.names() == ["add"]


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(make_spec())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_spec())


def test_unknown_tool_is_rejected() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="unknown tool"):
        registry.get("missing")


async def test_timeout_is_enforced() -> None:
    registry = ToolRegistry()
    registry.register(make_spec("slow", timeout_s=0.01, handler=slow))

    with pytest.raises(TimeoutError, match="timed out"):
        await registry.call("slow", AddInput(a=1, b=1))


async def test_wrong_payload_type_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(make_spec())

    with pytest.raises(TypeError, match="expects AddInput"):
        await registry.call("add", AddOutput(value=1))
