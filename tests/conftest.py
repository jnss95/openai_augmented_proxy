"""Pytest fixtures and configuration for openai-proxy tests."""

import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml

# Set test environment before importing app modules
os.environ["OPENAI_PROXY_CONFIG_DIR"] = "/tmp/openai_proxy_test_config"
os.environ["BASE_URL"] = "https://api.test.com/v1"
os.environ["API_KEY"] = "test-api-key"


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset all global state between tests."""
    # Reset module-level globals
    import openai_proxy.config as config_module
    import openai_proxy.models as models_module
    import openai_proxy.skills as skills_module
    import openai_proxy.handlers as handlers_module
    import openai_proxy.client as client_module
    import openai_proxy.chat_logger as chat_logger_module
    import openai_proxy.mcp_client as mcp_client_module
    
    # Clear cached settings
    config_module.get_settings.cache_clear()
    
    # Reset global instances
    models_module._registry = None
    skills_module._skills_loader = None
    client_module._client = None
    chat_logger_module._chat_logger = None
    mcp_client_module._mcp_client = None
    
    yield
    
    # Cleanup after test
    config_module.get_settings.cache_clear()
    models_module._registry = None
    skills_module._skills_loader = None
    client_module._client = None
    chat_logger_module._chat_logger = None
    mcp_client_module._mcp_client = None


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_dir(temp_dir: Path) -> Path:
    """Create a temporary config directory structure."""
    config_path = temp_dir / "config"
    (config_path / "models").mkdir(parents=True)
    (config_path / "mcp").mkdir(parents=True)
    (config_path / "skills").mkdir(parents=True)
    return config_path


@pytest.fixture
def sample_model_config() -> dict:
    """Return a sample model configuration."""
    return {
        "name": "test-assistant",
        "upstream_model": "gpt-4o-mini",
        "system_prompt": "You are a helpful test assistant.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Timezone name"
                            }
                        },
                        "required": []
                    }
                }
            }
        ],
        "mcp_servers": [],
        "skills": [],
        "include_global_skills": True,
        "owned_by": "test",
    }


@pytest.fixture
def model_config_file(config_dir: Path, sample_model_config: dict) -> Path:
    """Create a model config file and return its path."""
    model_file = config_dir / "models" / "test-assistant.yaml"
    with open(model_file, "w") as f:
        yaml.dump(sample_model_config, f)
    return model_file


@pytest.fixture
def sample_skill_content() -> str:
    """Return sample skill markdown content."""
    return """# Code Review Expert

You are an expert at reviewing code. Follow these guidelines:

1. Check for bugs and security issues
2. Suggest performance improvements
3. Ensure code follows best practices
"""


@pytest.fixture
def skill_file(config_dir: Path, sample_skill_content: str) -> Path:
    """Create a skill file and return its path."""
    skill_file = config_dir / "skills" / "code-review.md"
    with open(skill_file, "w") as f:
        f.write(sample_skill_content)
    return skill_file


@pytest.fixture
def mock_settings(config_dir: Path):
    """Create mock settings pointing to the temp config directory."""
    from openai_proxy.config import Settings
    
    settings = Settings(
        base_url="https://api.test.com/v1",
        api_key="test-api-key",
        host="127.0.0.1",
        port=8000,
        config_dir=str(config_dir),
        request_timeout=30.0,
    )
    
    with patch("openai_proxy.config.get_settings", return_value=settings):
        yield settings


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx async client."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    return mock_client


@pytest.fixture
def sample_chat_request() -> dict:
    """Return a sample chat completion request."""
    return {
        "model": "test-assistant",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }


@pytest.fixture
def sample_chat_response() -> dict:
    """Return a sample chat completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
    }


@pytest.fixture
def sample_tool_call_response() -> dict:
    """Return a sample chat response with tool calls."""
    return {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_current_time",
                                "arguments": '{"timezone": "UTC"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }


@pytest.fixture
def logs_dir(temp_dir: Path) -> Path:
    """Create a temporary logs directory."""
    logs_path = temp_dir / "logs"
    logs_path.mkdir(parents=True)
    return logs_path


@pytest.fixture
def mock_mcp_client():
    """Create a mock MCP client."""
    mock = AsyncMock()
    mock.get_connected_servers.return_value = []
    mock.get_tools.return_value = []
    mock.call_tool.return_value = {"result": "success"}
    return mock


@pytest.fixture
async def mock_mcp_client_context(mock_mcp_client):
    """Context manager for mocking MCP client."""
    with patch("openai_proxy.mcp_client.get_mcp_client", return_value=mock_mcp_client):
        yield mock_mcp_client
