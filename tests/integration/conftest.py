"""Fixtures for integration tests."""

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from tests.unit.server.conftest import server, state, test_app

__all__ = ["client", "server", "state", "test_app"]


@pytest_asyncio.fixture
async def client(test_app):
    # Session cookies are Secure, so they are only replayed over https.
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="https://test") as http:
        yield http
