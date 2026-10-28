"""Offline tests for token estimation.

tiktoken does not bundle its BPE vocabularies; asking it for ``cl100k_base``
downloads the vocabulary from the network on a cache miss. Tests must run
offline, so the positive-path test replaces the encoder lookup with a local
fake and never lets tiktoken reach ``tiktoken_ext.openai_public``.
"""

import pytest
import tiktoken

from eva_agent.adapters.target import tokens as tokens_module
from eva_agent.adapters.target.tokens import TokenEstimationError, estimate_tokens


class RecordingEncoding:
    """Minimal stand-in for ``tiktoken.Encoding`` with the interface we use."""

    def __init__(self, token_count: int) -> None:
        self._token_count = token_count
        self.encoded_texts: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded_texts.append(text)
        return list(range(self._token_count))


@pytest.fixture(autouse=True)
def clear_encoding_cache():
    tokens_module._encoding.cache_clear()
    yield
    tokens_module._encoding.cache_clear()


def test_estimate_tokens_returns_encoder_token_count(monkeypatch):
    fake_encoding = RecordingEncoding(token_count=7)
    requested_names: list[str] = []

    def fake_get_encoding(name: str) -> RecordingEncoding:
        requested_names.append(name)
        return fake_encoding

    monkeypatch.setattr(tokens_module.tiktoken, "get_encoding", fake_get_encoding)

    assert estimate_tokens("hello world", "cl100k_base") == 7
    assert requested_names == ["cl100k_base"]
    assert fake_encoding.encoded_texts == ["hello world"]


def test_unknown_encoding_rejected_without_network():
    # The real tiktoken registry rejects unknown names without any lookup, so
    # this path stays offline while still exercising the production code.
    with pytest.raises(ValueError, match="Unknown encoding"):
        tiktoken.get_encoding("definitely-not-an-encoding")

    with pytest.raises(
        TokenEstimationError,
        match="token estimation failed for encoding 'definitely-not-an-encoding'",
    ):
        estimate_tokens("hello", "definitely-not-an-encoding")
