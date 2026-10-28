"""OpenJudge chat model that never touches the network."""

from __future__ import annotations

import json
from typing import Any

from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.response import ChatResponse


class FakeChatModel(BaseChatModel):
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        super().__init__(model="fake", stream=False)
        self.response = response or {"reason": "ok", "score": 5}

    async def achat(self, *args: Any, **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            role="assistant",
            content=json.dumps(self.response, ensure_ascii=False),
            parsed=dict(self.response),
        )


class FailingChatModel(BaseChatModel):
    def __init__(self) -> None:
        super().__init__(model="fake", stream=False)

    async def achat(self, *args: Any, **kwargs: Any) -> ChatResponse:
        raise RuntimeError("model is down")
