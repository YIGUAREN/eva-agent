from pathlib import Path

import pytest
from pydantic import ValidationError

from eva_agent.config.settings import AigSettings, load_config


def test_aig_defaults_are_local_and_bounded() -> None:
    settings = AigSettings()
    assert settings.base_url == "http://127.0.0.1:8088"
    assert settings.scan.datasets == ["JailbreakPrompts-Tiny"]
    assert settings.scan.num_prompts == 20
    assert settings.scan.random_seed == 42


def test_aig_rejects_empty_scan_and_inverted_windows() -> None:
    with pytest.raises(ValidationError):
        AigSettings.model_validate({"scan": {"datasets": [], "num_prompts": 1, "random_seed": 1}})
    with pytest.raises(ValidationError):
        AigSettings(poll_initial_s=5, poll_max_s=2)
    with pytest.raises(ValidationError):
        AigSettings(min_wait_s=10, max_wait_s=5)


def test_example_config_loads_the_aig_section() -> None:
    config = load_config(Path("config/config.example.yaml"))
    assert config.aig is not None
    assert config.aig.base_url == "http://127.0.0.1:8088"
