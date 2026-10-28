import pytest

from eva_agent.adapters.target.tokens import TokenEstimationError, estimate_tokens


def test_estimate_tokens_returns_positive_count():
    assert estimate_tokens("hello world", "cl100k_base") >= 1


def test_unknown_encoding_rejected():
    with pytest.raises(TokenEstimationError, match="unknown token encoding"):
        estimate_tokens("hello", "not-an-encoding")
