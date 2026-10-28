import json
from typing import Literal

import pytest

from eva_agent.adapters.target import build_adapter
from eva_agent.adapters.target.base import TargetResponse, TargetResponseError
from eva_agent.adapters.target.http_json import HttpJsonTargetAdapter
from eva_agent.adapters.target.openai import OpenAITargetAdapter
from eva_agent.domain.target import TargetKind, TargetSpec, UsagePaths


def openai_target() -> TargetSpec:
    return TargetSpec(
        name="demo",
        kind=TargetKind.OPENAI_COMPATIBLE,
        provider="demo",
        model="demo-echo",
        base_url="http://127.0.0.1:8100/v1/",
    )


def json_target(method: Literal["GET", "POST"] = "POST") -> TargetSpec:
    template = {"messages": [{"role": "user", "content": "{{prompt}}"}]}
    if method == "GET":
        template = {"q": "{{prompt}}", "limit": 5}
    return TargetSpec(
        name="generic",
        kind=TargetKind.HTTP_JSON,
        provider="custom",
        model="custom-chat",
        url="https://example.com/search?lang=zh",
        method=method,
        request_template=template,
        response_path="$.answer",
        usage_paths=UsagePaths(input_tokens="$.usage.in", output_tokens="$.usage.out"),
    )


def test_openai_builds_chat_completions_request():
    request = OpenAITargetAdapter().build_request(
        openai_target(),
        "hello",
        headers={"Authorization": "Bearer sk-test"},
    )

    assert request.method == "POST"
    assert request.url == "http://127.0.0.1:8100/v1/chat/completions"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.body) == {
        "model": "demo-echo",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_openai_parses_content_and_usage():
    body = json.dumps(
        {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    ).encode()

    parsed = OpenAITargetAdapter().parse_response(openai_target(), TargetResponse(200, {}, body))

    assert parsed.text == "ok"
    assert parsed.input_tokens == 3
    assert parsed.output_tokens == 2


def test_openai_missing_usage_is_allowed():
    body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    parsed = OpenAITargetAdapter().parse_response(openai_target(), TargetResponse(200, {}, body))

    assert parsed.input_tokens is None
    assert parsed.output_tokens is None


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"choices": []}'])
def test_openai_malformed_response_rejected(body: bytes):
    with pytest.raises(TargetResponseError):
        OpenAITargetAdapter().parse_response(openai_target(), TargetResponse(200, {}, body))


def test_http_json_post_substitutes_nested_prompt():
    request = HttpJsonTargetAdapter().build_request(json_target("POST"), "hello", headers={})

    assert request.method == "POST"
    assert request.url == "https://example.com/search?lang=zh"
    assert json.loads(request.body) == {"messages": [{"role": "user", "content": "hello"}]}


def test_http_json_get_builds_query():
    request = HttpJsonTargetAdapter().build_request(json_target("GET"), "hello world", headers={})

    assert request.method == "GET"
    assert request.url == "https://example.com/search?lang=zh&q=hello+world&limit=5"
    assert request.body == b""


def test_http_json_parses_response_path_and_usage():
    body = b'{"answer": "yes", "usage": {"in": 4, "out": 2}}'

    parsed = HttpJsonTargetAdapter().parse_response(json_target(), TargetResponse(200, {}, body))

    assert parsed.text == "yes"
    assert parsed.input_tokens == 4
    assert parsed.output_tokens == 2


def test_http_json_missing_response_path_rejected():
    with pytest.raises(TargetResponseError, match="response_path"):
        HttpJsonTargetAdapter().parse_response(json_target(), TargetResponse(200, {}, b'{"x": 1}'))


def test_http_json_missing_usage_path_rejected():
    with pytest.raises(TargetResponseError, match="usage_paths"):
        HttpJsonTargetAdapter().parse_response(json_target(), TargetResponse(200, {}, b'{"answer": "yes"}'))


def test_factory_selects_by_kind():
    assert isinstance(build_adapter(openai_target()), OpenAITargetAdapter)
    assert isinstance(build_adapter(json_target()), HttpJsonTargetAdapter)
