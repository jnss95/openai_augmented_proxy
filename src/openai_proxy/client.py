"""Upstream API client for proxying requests."""

import json
from typing import Any, AsyncIterator

import httpx

from .config import get_settings
from .schemas import ChatCompletionRequest, ChatCompletionResponse


class UpstreamClient:
    """Client for communicating with the upstream OpenAI-compatible API."""

    def __init__(self):
        self.settings = get_settings()
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=httpx.Timeout(self.settings.request_timeout),
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        """Send a chat completion request to the upstream API."""
        payload = request.model_dump(exclude_none=True)

        response = await self.client.post(
            "/chat/completions",
            json=payload,
            **kwargs,
        )
        response.raise_for_status()

        return ChatCompletionResponse.model_validate(response.json())

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a streaming chat completion request to the upstream API."""
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = True

        async with self.client.stream(
            "POST",
            "/chat/completions",
            json=payload,
            **kwargs,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line[6:]  # Strip "data: " prefix
                elif line == "":
                    continue

    async def list_models(self) -> dict:
        """Fetch available models from the upstream API."""
        response = await self.client.get("/models")
        response.raise_for_status()
        return response.json()


# Global client instance
_client: UpstreamClient | None = None


def get_upstream_client() -> UpstreamClient:
    """Get the global upstream client."""
    global _client
    if _client is None:
        _client = UpstreamClient()
    return _client
