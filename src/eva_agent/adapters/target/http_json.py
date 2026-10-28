"""Generic JSON-over-HTTP target adapter with a documented path subset."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from eva_agent.adapters.target.base import (
    ParsedTargetResponse,
    TargetRequest,
    TargetResponse,
    TargetResponseError,
    load_json_body,
)
from eva_agent.domain.target import TargetSpec
from eva_agent.graders_dsl.paths import PathError, resolve_path


class HttpJsonTargetAdapter:
    def build_request(
        self,
        target: TargetSpec,
        prompt: str,
        *,
        headers: dict[str, str],
    ) -> TargetRequest:
        if not target.url or target.request_template is None or target.method is None:
            raise ValueError("http_json target is missing url, request_template, or method")

        template = _substitute(target.request_template, prompt)
        request_headers = dict(headers)
        if target.method == "POST":
            request_headers.setdefault("Content-Type", "application/json")
            return TargetRequest(
                method="POST",
                url=target.url,
                headers=request_headers,
                body=json.dumps(template, ensure_ascii=False).encode("utf-8"),
            )

        if not isinstance(template, dict):
            raise ValueError("GET request_template must be an object")
        for value in template.values():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError("GET request_template values must be strings or numbers")
        query = urlencode({key: str(value) for key, value in template.items()})
        parts = urlsplit(target.url)
        merged = "&".join(part for part in (parts.query, query) if part)
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, merged, parts.fragment))
        return TargetRequest(method="GET", url=url, headers=request_headers)

    def parse_response(self, target: TargetSpec, response: TargetResponse) -> ParsedTargetResponse:
        payload = load_json_body(response)
        if not target.response_path:
            raise TargetResponseError("http_json target is missing response_path")
        try:
            value = resolve_path(payload, target.response_path)
        except PathError as exc:
            raise TargetResponseError(f"response_path did not resolve: {exc}") from None

        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        input_tokens = None
        output_tokens = None
        if target.usage_paths is not None:
            if target.usage_paths.input_tokens:
                input_tokens = _token_value(payload, target.usage_paths.input_tokens, "input_tokens")
            if target.usage_paths.output_tokens:
                output_tokens = _token_value(payload, target.usage_paths.output_tokens, "output_tokens")

        return ParsedTargetResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens)


def _substitute(value: Any, prompt: str) -> Any:
    if isinstance(value, str):
        return value.replace("{{prompt}}", prompt)
    if isinstance(value, dict):
        return {key: _substitute(item, prompt) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, prompt) for item in value]
    return value


def _token_value(payload: Any, path: str, label: str) -> int:
    try:
        value = resolve_path(payload, path)
    except PathError as exc:
        raise TargetResponseError(f"usage_paths.{label} did not resolve: {exc}") from None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TargetResponseError(f"usage_paths.{label} is not a non-negative integer")
    return value
