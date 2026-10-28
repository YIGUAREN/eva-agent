import pytest
from pydantic import ValidationError

from eva_agent.config.settings import PricingSettings, SecuritySettings, TargetSettings


def test_target_settings_defaults():
    settings = TargetSettings()

    assert settings.targets_path == "config/targets.yaml"
    assert settings.timeout_s == 30
    assert settings.max_response_bytes == 1_048_576
    assert settings.token_encoding == "cl100k_base"


def test_pricing_and_security_defaults():
    assert PricingSettings().path == "config/pricing.yaml"
    assert SecuritySettings().policy_path == "config/security_policy.yaml"


def test_target_settings_reject_unknown_fields():
    with pytest.raises(ValidationError):
        TargetSettings.model_validate({"unknown": True})
