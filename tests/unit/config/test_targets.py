from pathlib import Path

import pytest

from eva_agent.config.settings import ConfigError
from eva_agent.config.targets import load_targets


def write_targets(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "targets.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_targets_injects_name_and_defaults_post(tmp_path: Path):
    path = write_targets(
        tmp_path,
        """
targets:
  demo:
    kind: openai_compatible
    provider: demo
    base_url: http://127.0.0.1:8100/v1
    model: demo-echo
  generic:
    kind: http_json
    provider: custom
    model: custom-chat
    url: https://example.com/chat
    request_template: {q: "{{prompt}}"}
    response_path: $.answer
""",
    )

    targets = load_targets(path)

    assert set(targets) == {"demo", "generic"}
    assert targets["demo"].name == "demo"
    assert targets["generic"].name == "generic"
    assert targets["generic"].method == "POST"


def test_load_targets_rejects_embedded_name(tmp_path: Path):
    path = write_targets(
        tmp_path,
        """
targets:
  demo:
    name: other
    kind: openai_compatible
    provider: demo
    base_url: http://127.0.0.1:8100/v1
    model: demo-echo
""",
    )

    with pytest.raises(ConfigError, match="must not define 'name'"):
        load_targets(path)


def test_load_targets_rejects_empty_file(tmp_path: Path):
    path = write_targets(tmp_path, "targets: {}\n")

    with pytest.raises(ConfigError, match="no targets"):
        load_targets(path)


def test_load_targets_wraps_validation_error(tmp_path: Path):
    path = write_targets(
        tmp_path,
        """
targets:
  broken:
    kind: http_json
    provider: custom
    model: custom-chat
    url: https://example.com/chat
    method: POST
""",
    )

    with pytest.raises(ConfigError, match="invalid target 'broken'"):
        load_targets(path)
