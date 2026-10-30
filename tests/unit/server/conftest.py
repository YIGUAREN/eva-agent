import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from eva_agent.server.state import AppState
from tests.support.server import build_test_app


@pytest_asyncio.fixture
async def server(tmp_path):
    app, state = await build_test_app(tmp_path)
    yield app, state
    await state.engine.dispose()


@pytest_asyncio.fixture
async def test_app(server):
    return server[0]


@pytest_asyncio.fixture
async def state(server) -> AppState:
    return server[1]


@pytest_asyncio.fixture
async def client(test_app):
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as http:
        yield http
