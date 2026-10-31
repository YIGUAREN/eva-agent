from pathlib import Path

import pytest
from pydantic import ValidationError

from eva_agent.config.settings import (
    ConfigError,
    UploadsSettings,
    WebSettings,
    load_config,
    require_sandbox_token,
    validate_sandbox_url,
)


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


def test_web_upload_defaults() -> None:
    web = WebSettings()
    assert web.upload_max_bytes == 52_428_800
    assert web.upload_timeout_s == 120.0
    assert web.sandbox_runner_url == "http://127.0.0.1:8090"
    assert web.sandbox_timeout_s == 120.0


def test_uploads_defaults() -> None:
    uploads = UploadsSettings()
    assert uploads.max_extracted_bytes == 536_870_912
    assert uploads.max_files == 10_000
    assert uploads.max_depth == 20


def test_upload_settings_reject_unknown_keys(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "public_base_url: http://localhost:8000\n"
        "retention_days: 30\n"
        "models: {}\n"
        "agent: {}\n"
        "storage: {url: 'sqlite+aiosqlite:///./x.db'}\n"
        "uploads: {max_files: 5, surprise: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="uploads"):
        load_config(path)


def test_require_sandbox_token(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_TOKEN", raising=False)
    with pytest.raises(ConfigError, match="SANDBOX_TOKEN"):
        require_sandbox_token()
    monkeypatch.setenv("SANDBOX_TOKEN", "t" * 32)
    assert require_sandbox_token() == "t" * 32


@pytest.mark.parametrize("value", ["", "not-a-url", "ftp://host", "http://"])
def test_validate_sandbox_url_rejects(value: str) -> None:
    with pytest.raises(ConfigError, match="sandbox_runner_url"):
        validate_sandbox_url(value)


def test_validate_sandbox_url_accepts_http() -> None:
    assert validate_sandbox_url("http://sandbox-runner:8090") == "http://sandbox-runner:8090"
