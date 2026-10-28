"""Deterministic OpenAI-compatible demo target."""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Eva Agent Demo Target")


def _count_tokens(text: str) -> int:
    """demo 目标只求确定性：按 UTF-8 字节数/4 估算，避免离线下载 tiktoken 词表。"""
    return max(1, len(text.encode("utf-8")) // 4)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "demo-echo"
    messages: list[Message] = Field(min_length=1)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> dict[str, Any]:
    delay_ms = max(0.0, _env_float("DEMO_TARGET_DELAY_MS", 0.0))
    if delay_ms:
        await asyncio.sleep(delay_ms / 1000)

    error_rate = min(1.0, max(0.0, _env_float("DEMO_TARGET_ERROR_RATE", 0.0)))
    if error_rate and random.random() < error_rate:
        raise HTTPException(status_code=500, detail="demo target failure")

    last_user = next(
        (message.content for message in reversed(request.messages) if message.role == "user"),
        "",
    )
    text = f"demo answer: {last_user}"
    prompt_tokens = _count_tokens("\n".join(message.content for message in request.messages))
    completion_tokens = _count_tokens(text)

    return {
        "id": "demo-completion",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
