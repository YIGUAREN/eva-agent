from pathlib import Path

import pytest
from pydantic import ValidationError

from eva_agent.config.settings import (
    ConfigError,
    MissingSecretError,
    ModelConfig,
    load_config,
    resolve_api_key,
)

CONFIG_YAML = """
public_base_url: http://localhost:8000
retention_days: 30
storage:
  url: sqlite+aiosqlite:///./data/eva.db
agent:
  max_steps: 12
models:
  agent:
    provider: openai-compatible
    model: deepseek-chat
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
  judge:
    provider: openai-compatible
    model: qwen-plus
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key_env: DASHSCOPE_API_KEY
  attack:
    provider: openai-compatible
    model: deepseek-chat
    base_url: https://api.deepseek.com/v1
    api_key_env: DEEPSEEK_API_KEY
"""


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_parses_yaml(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, CONFIG_YAML))

    assert config.public_base_url == "http://localhost:8000"
    assert config.agent.max_steps == 12
    assert config.models.agent.model == "deepseek-chat"
    assert config.models.judge.api_key_env == "DASHSCOPE_API_KEY"


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_invalid_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write_config(tmp_path, "models: ["))


def test_load_config_wraps_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, CONFIG_YAML.replace("max_steps: 12", "max_steps: 0")))


def test_litellm_requires_base_url() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="litellm", model="m", base_url=None, api_key_env="MY_KEY")


def test_openai_compatible_requires_base_url() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai-compatible", model="m", base_url=None, api_key_env="MY_KEY")


def test_validation_error_does_not_echo_secret(tmp_path: Path) -> None:
    text = CONFIG_YAML.replace(
        "    api_key_env: DEEPSEEK_API_KEY\n",
        "    api_key_env: DEEPSEEK_API_KEY\n    api_key: sk-live-super-secret\n",
        1,
    )
    assert "sk-live-super-secret" in text

    with pytest.raises(ConfigError) as excinfo:
        load_config(write_config(tmp_path, text))

    message = str(excinfo.value)
    assert "models.agent.api_key" in message
    assert "sk-live-super-secret" not in message


def test_resolve_api_key_reads_mapping() -> None:
    model = ModelConfig(
        provider="openai-compatible",
        model="m",
        base_url="https://example.test/v1",
        api_key_env="MY_KEY",
    )

    assert resolve_api_key(model, {"MY_KEY": "sk-test"}) == "sk-test"


def test_resolve_api_key_missing_raises() -> None:
    model = ModelConfig(
        provider="openai-compatible",
        model="m",
        base_url="https://example.test/v1",
        api_key_env="MY_KEY",
    )

    with pytest.raises(MissingSecretError, match="MY_KEY"):
        resolve_api_key(model, {})
