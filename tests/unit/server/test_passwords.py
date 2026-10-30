import pytest

from eva_agent.server.passwords import hash_password, needs_rehash, verify_password


def test_hash_and_verify_round_trip() -> None:
    encoded = hash_password("correct horse battery")

    assert encoded.startswith("$argon2id$")
    assert verify_password("correct horse battery", encoded) is True
    assert verify_password("wrong password", encoded) is False


def test_hash_is_salted() -> None:
    assert hash_password("correct horse battery") != hash_password("correct horse battery")


def test_short_password_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 12"):
        hash_password("short")


def test_verify_rejects_a_malformed_hash() -> None:
    assert verify_password("anything", "not-a-hash") is False


def test_needs_rehash_is_false_for_a_fresh_hash() -> None:
    assert needs_rehash(hash_password("correct horse battery")) is False


def test_script_refuses_mismatched_confirmation(monkeypatch, capsys) -> None:
    from scripts.hash_password import main

    answers = iter(["correct horse battery", "different password"])
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))

    assert main() == 1
    assert "do not match" in capsys.readouterr().err
