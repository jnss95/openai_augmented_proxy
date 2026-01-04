"""Tests for client.py - Upstream API client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openai_proxy.client import UpstreamClient, get_upstream_client
from openai_proxy.schemas import ChatCompletionRequest, ChatCompletionResponse, Message


class TestUpstreamClient:
    """Tests for UpstreamClient class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.base_url = "https://api.test.com/v1"
        settings.api_key = "test-api-key"
        settings.request_timeout = 30.0
        return settings

    @pytest.fixture
    def client(self, mock_settings):
        """Create an UpstreamClient with mock settings."""
        with patch("openai_proxy.client.get_settings", return_value=mock_settings):
            yield UpstreamClient()

    def test_client_initialization(self, client, mock_settings):
        """Test client initialization."""
        assert client.settings == mock_settings
        assert client._client is None  # Lazy initialization

    def test_client_property_creates_client(self, client, mock_settings):
        """Test that client property creates httpx client."""
        http_client = client.client
        
        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)

    def test_client_property_reuses_client(self, client):
        """Test that client property reuses existing client."""
        client1 = client.client
        client2 = client.client
        
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_close_client(self, client):
        """Test closing the client."""
        # Access client to create it
        _ = client.client
        assert client._client is not None
        
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialized(self, client):
        """Test closing when client was never created."""
        # Should not raise
        await client.close()
        assert client._client is None


class TestUpstreamClientChatCompletion:
    """Tests for chat completion methods."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

    @pytest.fixture
    def sample_request(self):
        """Create a sample chat request."""
        return ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hi")],
            temperature=0.7,
        )

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, mock_response, sample_request):
        """Test successful chat completion."""
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response
        mock_http_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_http_response
        mock_http_client.is_closed = False
        
        with patch("openai_proxy.client.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "https://api.test.com/v1"
            mock_settings.return_value.api_key = "test-key"
            mock_settings.return_value.request_timeout = 30.0
            
            client = UpstreamClient()
            client._client = mock_http_client
            
            response = await client.chat_completion(sample_request)
        
        assert isinstance(response, ChatCompletionResponse)
        assert response.id == "chatcmpl-123"
        assert response.choices[0].message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_completion_sends_correct_payload(self, mock_response, sample_request):
        """Test that chat completion sends correct payload."""
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response
        mock_http_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_http_response
        mock_http_client.is_closed = False
        
        with patch("openai_proxy.client.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "https://api.test.com/v1"
            mock_settings.return_value.api_key = "test-key"
            mock_settings.return_value.request_timeout = 30.0
            
            client = UpstreamClient()
            client._client = mock_http_client
            
            await client.chat_completion(sample_request)
        
        # Check the call was made with correct endpoint
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == "/chat/completions"
        
        # Check payload
        payload = call_args[1]["json"]
        assert payload["model"] == "gpt-4"
        assert payload["temperature"] == 0.7


class TestUpstreamClientStreaming:
    """Tests for streaming chat completion."""

    @pytest.mark.asyncio
    async def test_chat_completion_stream(self):
        """Test streaming chat completion."""
        # Mock streaming response
        async def mock_aiter_lines():
            yield "data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}"
            yield "data: {\"id\": \"1\", \"choices\": [{\"delta\": {\"content\": \" world\"}}]}"
            yield "data: [DONE]"
            yield ""
        
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_http_client = MagicMock()
        mock_http_client.stream.return_value = mock_response
        mock_http_client.is_closed = False
        
        with patch("openai_proxy.client.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "https://api.test.com/v1"
            mock_settings.return_value.api_key = "test-key"
            mock_settings.return_value.request_timeout = 30.0
            
            client = UpstreamClient()
            client._client = mock_http_client
            
            request = ChatCompletionRequest(
                model="gpt-4",
                messages=[Message(role="user", content="Hi")],
            )
            
            chunks = []
            async for chunk in client.chat_completion_stream(request):
                chunks.append(chunk)
        
        # Should have received the content chunks (without "data: " prefix)
        assert len(chunks) >= 2
        assert "Hello" in chunks[0]
        assert "world" in chunks[1]


class TestUpstreamClientListModels:
    """Tests for list models method."""

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4", "created": 1700000000, "owned_by": "openai"},
                {"id": "gpt-3.5-turbo", "created": 1699000000, "owned_by": "openai"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        mock_http_client.get.return_value = mock_response
        mock_http_client.is_closed = False
        
        with patch("openai_proxy.client.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "https://api.test.com/v1"
            mock_settings.return_value.api_key = "test-key"
            mock_settings.return_value.request_timeout = 30.0
            
            client = UpstreamClient()
            client._client = mock_http_client
            
            result = await client.list_models()
        
        assert "data" in result
        assert len(result["data"]) == 2
        assert result["data"][0]["id"] == "gpt-4"


class TestGetUpstreamClient:
    """Tests for get_upstream_client function."""

    def test_get_upstream_client_creates_singleton(self):
        """Test that get_upstream_client creates singleton."""
        import openai_proxy.client as client_module
        client_module._client = None
        
        with patch("openai_proxy.client.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "https://api.test.com/v1"
            mock_settings.return_value.api_key = "test-key"
            mock_settings.return_value.request_timeout = 30.0
            
            client1 = get_upstream_client()
            client2 = get_upstream_client()
            
            assert client1 is client2

    def test_get_upstream_client_initializes_correctly(self):
        """Test that get_upstream_client initializes client correctly."""
        import openai_proxy.client as client_module
        client_module._client = None
        
        with patch("openai_proxy.client.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "https://api.openai.com/v1"
            mock_settings.return_value.api_key = "sk-test"
            mock_settings.return_value.request_timeout = 60.0
            
            client = get_upstream_client()
            
            assert isinstance(client, UpstreamClient)
            assert client.settings.base_url == "https://api.openai.com/v1"


class TestClientErrorHandling:
    """Tests for client error handling."""

    @pytest.mark.asyncio
    async def test_http_error_propagates(self):
        """Test that HTTP errors propagate correctly."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=MagicMock(status_code=500)
        )
        
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        mock_http_client.is_closed = False
        
        with patch("openai_proxy.client.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "https://api.test.com/v1"
            mock_settings.return_value.api_key = "test-key"
            mock_settings.return_value.request_timeout = 30.0
            
            client = UpstreamClient()
            client._client = mock_http_client
            
            request = ChatCompletionRequest(
                model="gpt-4",
                messages=[Message(role="user", content="Hi")],
            )
            
            with pytest.raises(httpx.HTTPStatusError):
                await client.chat_completion(request)

    @pytest.mark.asyncio
    async def test_recreates_client_if_closed(self):
        """Test that client is recreated if closed."""
        with patch("openai_proxy.client.get_settings") as mock_settings:
            mock_settings.return_value.base_url = "https://api.test.com/v1"
            mock_settings.return_value.api_key = "test-key"
            mock_settings.return_value.request_timeout = 30.0
            
            client = UpstreamClient()
            
            # Create a closed mock client
            mock_closed_client = MagicMock()
            mock_closed_client.is_closed = True
            client._client = mock_closed_client
            
            # Access client property should create new client
            new_client = client.client
            
            assert new_client is not mock_closed_client
            assert isinstance(new_client, httpx.AsyncClient)
