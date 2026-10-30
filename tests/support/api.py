from httpx import AsyncClient

from tests.support.server import TEST_PASSWORD

PASTE_PAYLOAD = {
    "input": {"kind": "paste", "paste": {"query": "what is 2+2", "response": "4"}},
    "budget": {"max_cost_usd": 1.0, "max_wall_clock_s": 60.0, "max_target_requests": 1},
}


async def login(client: AsyncClient, password: str = TEST_PASSWORD) -> str:
    response = await client.post("/api/auth/login", json={"password": password})
    assert response.status_code == 200, response.text
    return response.json()["csrf_token"]


def csrf_headers(csrf: str) -> dict[str, str]:
    return {"X-CSRF-Token": csrf}
