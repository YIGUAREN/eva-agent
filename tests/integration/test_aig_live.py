"""Opt-in smoke test for a running local AIG stack. Run with: pytest -m aig."""

import os

import httpx
import pytest

pytestmark = pytest.mark.aig


async def test_aig_server_is_reachable() -> None:
    base_url = os.getenv("EVA_AIG_BASE_URL", "http://127.0.0.1:8088")
    async with httpx.AsyncClient(base_url=base_url, timeout=5) as client:
        response = await client.get("/")
    assert response.status_code < 500
