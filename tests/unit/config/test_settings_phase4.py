from pathlib import Path

import pytest
from pydantic import ValidationError

from eva_agent.config.settings import ConfigError, WebSettings, load_config


def test_web_settings_defaults() -> None:
    settings = WebSettings()

    assert settings.session_ttl_s == 43_200
    assert settings.session_absolute_ttl_s == 604_800
    assert settings.cookie_secure is True
    assert settings.max_concurrent_runs == 2
    assert settings.poll_interval_s == 0.5


def test_web_settings_reject_zero_concurrency() -> None:
    with pytest.raises(ValidationError):
        WebSettings(max_concurrent_runs=0)


def test_example_config_loads_the_web_block() -> None:
    config = load_config(Path("config/config.example.yaml"))

    assert config.web.max_concurrent_runs == 2
    assert config.web.session_ttl_s == 43_200


def test_unknown_web_key_is_rejected(tmp_path: Path) -> None:
    source = Path("config/config.example.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "config.yaml"
    bad.write_text(source + "\nweb:\n  unknown_key: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(bad)
