from fastapi.testclient import TestClient

from demo_target.app import app

client = TestClient(app)


def chat_body(content: str = "hi") -> dict:
    return {
        "model": "demo-echo",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": content},
        ],
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_completion_echoes_last_user_message():
    response = client.post("/v1/chat/completions", json=chat_body("reset password"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": "demo answer: reset password"}
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["usage"]["completion_tokens"] > 0
    assert payload["usage"]["total_tokens"] == (
        payload["usage"]["prompt_tokens"] + payload["usage"]["completion_tokens"]
    )


def test_error_rate_one_returns_500(monkeypatch):
    monkeypatch.setenv("DEMO_TARGET_ERROR_RATE", "1")

    response = client.post("/v1/chat/completions", json=chat_body())

    assert response.status_code == 500


def test_empty_messages_rejected():
    response = client.post("/v1/chat/completions", json={"model": "demo-echo", "messages": []})

    assert response.status_code == 422
