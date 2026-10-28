import pytest
from pydantic import ValidationError

from eva_agent.domain.run import InputKind, PastedInput, RunInput


def test_target_input_requires_ref_and_prompts():
    with pytest.raises(ValidationError, match="target input requires 'target_ref'"):
        RunInput(kind=InputKind.TARGET, target_prompts=["q"])
    with pytest.raises(ValidationError, match="target input requires at least one prompt"):
        RunInput(kind=InputKind.TARGET, target_ref="demo")


def test_non_target_input_rejects_prompts():
    with pytest.raises(ValidationError, match="'target_prompts' is only valid for target input"):
        RunInput(kind=InputKind.PASTE, paste=PastedInput(query="q"), target_prompts=["q"])


def test_target_prompts_reject_blank_and_oversized():
    with pytest.raises(ValidationError, match="prompt must not be blank"):
        RunInput(kind=InputKind.TARGET, target_ref="demo", target_prompts=["  "])
    with pytest.raises(ValidationError, match="prompt exceeds 16384 bytes"):
        RunInput(kind=InputKind.TARGET, target_ref="demo", target_prompts=["x" * 16385])


def test_target_prompts_reject_too_many():
    with pytest.raises(ValidationError, match="at most 100 prompts"):
        RunInput(kind=InputKind.TARGET, target_ref="demo", target_prompts=["q"] * 101)
