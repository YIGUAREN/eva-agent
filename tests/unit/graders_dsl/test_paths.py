import pytest

from eva_agent.graders_dsl.paths import PathError, parse_path, resolve_path


def test_parse_path_supports_keys_and_indices():
    assert parse_path("$.choices[0].message.content") == ["choices", 0, "message", "content"]
    assert parse_path("$") == []


def test_resolve_path_returns_root_and_nested_value():
    data = {"choices": [{"message": {"content": "hi"}}]}
    assert resolve_path(data, "$") == data
    assert resolve_path(data, "$.choices[0].message.content") == "hi"


@pytest.mark.parametrize("path", ["choices", "$.", "$.items[x]", "$.items[-1]"])
def test_parse_path_rejects_invalid_syntax(path: str):
    with pytest.raises(PathError):
        parse_path(path)


def test_resolve_path_rejects_missing_and_wrong_type():
    with pytest.raises(PathError):
        resolve_path({}, "$.missing")
    with pytest.raises(PathError):
        resolve_path({"items": {}}, "$.items[0]")
