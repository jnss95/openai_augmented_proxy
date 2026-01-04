"""Integration tests for app.py - FastAPI application."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from openai_proxy.schemas import ChatCompletionResponse, Choice, Message, Usage


@pytest.fixture
def test_config_dir():
    """Create a temporary config directory with test models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        
        # Create directories
        (config_dir / "models").mkdir()
        (config_dir / "mcp").mkdir()
        (config_dir / "skills").mkdir()
        (config_dir / "logs").mkdir()
        
        # Create test model config
        model_config = {
            "name": "test-assistant",
            "upstream_model": "gpt-4o-mini",
            "system_prompt": "You are a test assistant.",
            "tools": [],
            "mcp_servers": [],
        }
        
        with open(config_dir / "models" / "test.yaml", "w") as f:
            yaml.dump(model_config, f)
        
        yield config_dir


@pytest.fixture
def mock_upstream_client():
    """Create a mock upstream client."""
    mock_client = AsyncMock()
    
    # Mock list_models response
    mock_client.list_models.return_value = {
        "data": [
            {"id": "gpt-4", "created": 1700000000, "owned_by": "openai"},
            {"id": "gpt-3.5-turbo", "created": 1699000000, "owned_by": "openai"},
        ]
    }
    
    # Mock chat_completion response
    mock_response = ChatCompletionResponse(
        id="chatcmpl-test",
        created=1700000000,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="Hello! How can I help you?"),
                finish_reason="stop"
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    )
    mock_client.chat_completion.return_value = mock_response
    
    return mock_client


@pytest.fixture
def client(test_config_dir, mock_upstream_client):
    """Create a test client with mocked dependencies."""
    # Reset global state
    import openai_proxy.config as config_module
    import openai_proxy.models as models_module
    import openai_proxy.skills as skills_module
    import openai_proxy.client as client_module
    import openai_proxy.chat_logger as chat_logger_module
    import openai_proxy.mcp_client as mcp_client_module
    
    config_module.get_settings.cache_clear()
    models_module._registry = None
    skills_module._skills_loader = None
    client_module._client = None
    chat_logger_module._chat_logger = None
    mcp_client_module._mcp_client = None
    
    # Create mock settings
    from openai_proxy.config import Settings
    mock_settings = Settings(
        base_url="https://api.test.com/v1",
        api_key="test-api-key",
        config_dir=str(test_config_dir),
    )
    
    # Pre-load the model registry with the test config
    from openai_proxy.models import ModelRegistry
    registry = ModelRegistry()
    registry.load_from_directory(test_config_dir / "models")
    models_module._registry = registry
    
    # Mock MCP client
    mock_mcp = MagicMock()
    mock_mcp.get_connected_servers.return_value = []
    mock_mcp.get_tools.return_value = []
    
    # Patch at module level before importing app
    with patch.object(config_module, "get_settings", return_value=mock_settings):
        with patch("openai_proxy.client.get_upstream_client", return_value=mock_upstream_client):
            with patch("openai_proxy.proxy.get_upstream_client", return_value=mock_upstream_client):
                with patch("openai_proxy.mcp_client.get_mcp_client", return_value=mock_mcp):
                    with patch("openai_proxy.mcp_client.initialize_mcp_client", return_value=mock_mcp):
                        with patch("openai_proxy.mcp_client.shutdown_mcp_client", return_value=None):
                            with patch("openai_proxy.proxy.get_mcp_tools_for_model", return_value=[]):
                                from openai_proxy.app import app
                                yield TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestModelsEndpoint:
    """Tests for models listing endpoint."""

    def test_list_models(self, client, mock_upstream_client):
        """Test listing models."""
        response = client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["object"] == "list"
        assert "data" in data
        
        # Should include both upstream and augmented models
        model_ids = [m["id"] for m in data["data"]]
        assert "test-assistant" in model_ids  # Our augmented model

    def test_get_specific_model(self, client):
        """Test getting a specific augmented model."""
        response = client.get("/v1/models/test-assistant")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test-assistant"
        assert data["object"] == "model"

    def test_get_nonexistent_model(self, client):
        """Test getting a non-existent model."""
        response = client.get("/v1/models/nonexistent-model")
        
        assert response.status_code == 404


class TestChatCompletionsEndpoint:
    """Tests for chat completions endpoint."""

    def test_chat_completion_augmented_model(self, client):
        """Test chat completion with augmented model."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model"] == "test-assistant"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completion_with_temperature(self, client):
        """Test chat completion with temperature parameter."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.7,
                "max_tokens": 100,
            }
        )
        
        assert response.status_code == 200

    def test_chat_completion_passthrough(self, client, mock_upstream_client):
        """Test chat completion passthrough for unknown models."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        
        assert response.status_code == 200
        # Should call upstream directly
        mock_upstream_client.chat_completion.assert_called()

    def test_chat_completion_with_system_message(self, client):
        """Test chat completion with system message."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello!"}
                ],
            }
        )
        
        assert response.status_code == 200


class TestChatCompletionsStreaming:
    """Tests for streaming chat completions."""

    def test_streaming_request(self, client):
        """Test streaming chat completion request."""
        # Note: TestClient handles streaming differently
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            }
        )
        
        assert response.status_code == 200
        # Streaming response should have specific content type
        assert "text/event-stream" in response.headers.get("content-type", "")


class TestAdminEndpoints:
    """Tests for admin endpoints."""

    def test_reload_config(self, client, test_config_dir):
        """Test config reload endpoint."""
        import openai_proxy.mcp_client as mcp_client_module
        
        # Create a fresh mock MCP client for the reload
        mock_mcp = MagicMock()
        mock_mcp.get_connected_servers.return_value = []
        
        with patch("openai_proxy.app.shutdown_mcp_client", return_value=None):
            with patch("openai_proxy.app.initialize_mcp_client", return_value=mock_mcp):
                response = client.post("/admin/reload")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "skills_loaded" in data


class TestRequestValidation:
    """Tests for request validation."""

    def test_missing_model(self, client):
        """Test request without model field."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        
        assert response.status_code == 422  # Validation error

    def test_missing_messages(self, client):
        """Test request without messages field."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
            }
        )
        
        assert response.status_code == 422  # Validation error

    def test_invalid_message_role(self, client):
        """Test request with invalid message role."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
                "messages": [{"role": "invalid", "content": "Hi"}],
            }
        )
        
        assert response.status_code == 422


class TestToolsInRequest:
    """Tests for handling tools in requests."""

    def test_request_with_client_tools(self, client):
        """Test request with client-provided tools."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"}
                                },
                                "required": ["location"]
                            }
                        }
                    }
                ]
            }
        )
        
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""

    def test_upstream_error_handling(self, client, mock_upstream_client):
        """Test handling of upstream errors."""
        import httpx
        
        # Make upstream raise an error
        mock_upstream_client.chat_completion.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500, text="Server error")
        )
        
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        
        # Should return 500 error
        assert response.status_code == 500


class TestResponseFormat:
    """Tests for response format compliance."""

    def test_response_has_required_fields(self, client):
        """Test that response has all required OpenAI fields."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        
        # Choice fields
        assert len(data["choices"]) > 0
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice
        
        # Message fields
        assert "role" in choice["message"]
        assert "content" in choice["message"]

    def test_response_object_type(self, client):
        """Test that response has correct object type."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-assistant",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        
        data = response.json()
        assert data["object"] == "chat.completion"


class TestTemplateEvalEndpoint:
    """Tests for template evaluation endpoint."""

    def test_eval_simple_template(self, client):
        """Test evaluating a simple template."""
        response = client.post(
            "/admin/template/eval",
            json={
                "template": "Hello {{ name }}!",
                "variables": {"name": "World"},
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"] == "Hello World!"
        assert data["error"] is None

    def test_eval_template_with_tool_call(self, client):
        """Test evaluating a template with tool call."""
        response = client.post(
            "/admin/template/eval",
            json={
                "template": '{{ tool("calculator", expression="2+2") }}',
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "4" in data["result"]  # Calculator returns {"result": 4, ...}

    def test_eval_template_with_conditionals(self, client):
        """Test evaluating a template with conditionals."""
        response = client.post(
            "/admin/template/eval",
            json={
                "template": "{% if debug %}Debug ON{% else %}Debug OFF{% endif %}",
                "variables": {"debug": True},
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result"] == "Debug ON"

    def test_eval_template_with_loop(self, client):
        """Test evaluating a template with loops."""
        response = client.post(
            "/admin/template/eval",
            json={
                "template": "{% for i in items %}{{ i }} {% endfor %}",
                "variables": {"items": ["a", "b", "c"]},
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "a" in data["result"]
        assert "b" in data["result"]
        assert "c" in data["result"]

    def test_eval_invalid_template(self, client):
        """Test evaluating an invalid template returns error."""
        response = client.post(
            "/admin/template/eval",
            json={
                "template": "{% if unclosed",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is False
        assert data["result"] is None
        assert data["error"] is not None

    def test_get_variables(self, client):
        """Test getting available variables."""
        response = client.get("/admin/template/variables")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "variables" in data
        assert isinstance(data["variables"], dict)


class TestMCPEndpoints:
    """Tests for MCP server endpoints."""

    @pytest.fixture
    def mcp_client_with_servers(self, test_config_dir, mock_upstream_client):
        """Create a test client with mocked MCP servers."""
        # Reset global state
        import openai_proxy.config as config_module
        import openai_proxy.models as models_module
        import openai_proxy.skills as skills_module
        import openai_proxy.client as client_module
        import openai_proxy.chat_logger as chat_logger_module
        import openai_proxy.mcp_client as mcp_client_module
        
        config_module.get_settings.cache_clear()
        models_module._registry = None
        skills_module._skills_loader = None
        client_module._client = None
        chat_logger_module._chat_logger = None
        mcp_client_module._mcp_client = None
        
        from openai_proxy.config import Settings
        mock_settings = Settings(
            base_url="https://api.test.com/v1",
            api_key="test-api-key",
            config_dir=str(test_config_dir),
        )
        
        from openai_proxy.models import ModelRegistry
        registry = ModelRegistry()
        registry.load_from_directory(test_config_dir / "models")
        models_module._registry = registry
        
        # Create a mock MCP client with servers
        mock_mcp = MagicMock()
        mock_mcp.get_available_servers.return_value = ["test-server", "another-server"]
        mock_mcp.get_connected_servers.return_value = ["test-server"]
        mock_mcp.get_tools.return_value = []
        
        # Mock server info
        mock_mcp.get_all_servers_info.return_value = [
            {
                "name": "test-server",
                "type": "stdio",
                "description": "Test MCP server",
                "connected": True,
                "tools_count": 2,
                "config": {"command": "test-cmd", "args": [], "url": None},
            },
            {
                "name": "another-server",
                "type": "sse",
                "description": "Another server",
                "connected": False,
                "tools_count": 0,
                "config": {"command": None, "args": [], "url": "http://example.com"},
            },
        ]
        
        mock_mcp.get_server_info.side_effect = lambda name: {
            "test-server": {
                "name": "test-server",
                "type": "stdio",
                "description": "Test MCP server",
                "connected": True,
                "tools_count": 2,
                "config": {"command": "test-cmd", "args": [], "url": None},
            },
            "another-server": {
                "name": "another-server",
                "type": "sse",
                "description": "Another server",
                "connected": False,
                "tools_count": 0,
                "config": {"command": None, "args": [], "url": "http://example.com"},
            },
        }.get(name)
        
        mock_mcp.get_server_tools.side_effect = lambda name: {
            "test-server": [
                {
                    "name": "test_tool",
                    "full_name": "mcp_test-server_test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg1": {"type": "string", "description": "First argument"},
                        },
                        "required": ["arg1"],
                    },
                },
                {
                    "name": "another_tool",
                    "full_name": "mcp_test-server_another_tool",
                    "description": "Another tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
            "another-server": [],
        }.get(name, [])
        
        with patch.object(config_module, "get_settings", return_value=mock_settings):
            with patch("openai_proxy.client.get_upstream_client", return_value=mock_upstream_client):
                with patch("openai_proxy.proxy.get_upstream_client", return_value=mock_upstream_client):
                    with patch("openai_proxy.mcp_client.get_mcp_client", return_value=mock_mcp):
                        with patch("openai_proxy.app.get_mcp_client", return_value=mock_mcp):
                            with patch("openai_proxy.mcp_client.initialize_mcp_client", return_value=mock_mcp):
                                with patch("openai_proxy.mcp_client.shutdown_mcp_client", return_value=None):
                                    with patch("openai_proxy.proxy.get_mcp_tools_for_model", return_value=[]):
                                        from openai_proxy.app import app
                                        yield TestClient(app, raise_server_exceptions=False)

    def test_list_mcp_servers(self, mcp_client_with_servers):
        """Test listing all MCP servers."""
        response = mcp_client_with_servers.get("/admin/mcp/servers")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "servers" in data
        assert "total" in data
        assert "connected" in data
        assert data["total"] == 2
        assert data["connected"] == 1

    def test_get_mcp_server(self, mcp_client_with_servers):
        """Test getting a specific MCP server."""
        response = mcp_client_with_servers.get("/admin/mcp/servers/test-server")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "test-server"
        assert data["type"] == "stdio"
        assert data["connected"] is True

    def test_get_nonexistent_mcp_server(self, mcp_client_with_servers):
        """Test getting a nonexistent MCP server."""
        response = mcp_client_with_servers.get("/admin/mcp/servers/nonexistent")
        
        assert response.status_code == 404

    def test_list_mcp_server_tools(self, mcp_client_with_servers):
        """Test listing tools for a specific MCP server."""
        response = mcp_client_with_servers.get("/admin/mcp/servers/test-server/tools")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["server"] == "test-server"
        assert "tools" in data
        assert data["total"] == 2
        
        tool_names = [t["name"] for t in data["tools"]]
        assert "test_tool" in tool_names
        assert "another_tool" in tool_names

    def test_list_tools_disconnected_server(self, mcp_client_with_servers):
        """Test listing tools for a disconnected server."""
        response = mcp_client_with_servers.get("/admin/mcp/servers/another-server/tools")
        
        assert response.status_code == 503

    def test_get_mcp_tool(self, mcp_client_with_servers):
        """Test getting a specific tool."""
        response = mcp_client_with_servers.get("/admin/mcp/servers/test-server/tools/test_tool")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "test_tool"
        assert data["description"] == "A test tool"
        assert "parameters" in data

    def test_get_nonexistent_tool(self, mcp_client_with_servers):
        """Test getting a nonexistent tool."""
        response = mcp_client_with_servers.get("/admin/mcp/servers/test-server/tools/nonexistent")
        
        assert response.status_code == 404

    def test_list_all_mcp_tools(self, mcp_client_with_servers):
        """Test listing all MCP tools."""
        response = mcp_client_with_servers.get("/admin/mcp/tools")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "servers" in data
        assert "total_tools" in data
        assert "connected_servers" in data
        assert data["total_tools"] == 2
        assert "test-server" in data["servers"]
