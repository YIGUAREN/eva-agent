"""OpenAI-compatible chat completions adapter."""

from __future__ import annotations

import json

from eva_agent.adapters.target.base import (
    ParsedTargetResponse,
    TargetRequest,
    TargetResponse,
    TargetResponseError,
    int_or_none,
    load_json_body,
)
from eva_agent.domain.target import TargetSpec


class OpenAITargetAdapter:
    def build_request(
        self,
        target: TargetSpec,
        prompt: str,
        *,
        headers: dict[str, str],
    ) -> TargetRequest:
        if not target.base_url:
            raise ValueError("openai_compatible target requires base_url")
        payload = {
            "model": target.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        request_headers = dict(headers)
        request_headers["Content-Type"] = "application/json"
        return TargetRequest(
            method="POST",
            url=f"{target.base_url.rstrip('/')}/chat/completions",
            headers=request_headers,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )

    def parse_response(self, target: TargetSpec, response: TargetResponse) -> ParsedTargetResponse:
        payload = load_json_body(response)
        if not isinstance(payload, dict):
            raise TargetResponseError("openai response root must be an object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TargetResponseError("openai response is missing choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise TargetResponseError("openai response is missing choices[0].message.content")

        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return ParsedTargetResponse(
            text=content,
            input_tokens=int_or_none(usage.get("prompt_tokens")),
            output_tokens=int_or_none(usage.get("completion_tokens")),
        )
