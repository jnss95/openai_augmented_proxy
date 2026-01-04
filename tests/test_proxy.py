"""Tests for proxy.py - Proxy logic for handling chat completions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_proxy.models import ModelConfig, Tool, ToolFunction
from openai_proxy.mcp_client import MCPTool
from openai_proxy.proxy import (
    build_system_prompt,
    execute_proxy_tool,
    get_mcp_server_filters,
    get_mcp_server_names,
    get_proxy_tool_names,
    matches_pattern,
    merge_tools,
    prepend_system_prompt,
)
from openai_proxy.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    ToolFunctionSchema,
    ToolSchema,
    Usage,
)


class TestMatchesPattern:
    """Tests for matches_pattern function."""

    def test_exact_match(self):
        """Test exact string match."""
        assert matches_pattern("get_time", "get_time") is True
        assert matches_pattern("get_time", "get_weather") is False

    def test_wildcard_suffix(self):
        """Test wildcard suffix pattern."""
        assert matches_pattern("mcp_server_tool", "mcp_server_*") is True
        assert matches_pattern("mcp_other_tool", "mcp_server_*") is False

    def test_wildcard_prefix(self):
        """Test wildcard prefix pattern."""
        assert matches_pattern("get_current_time", "*_time") is True
        assert matches_pattern("get_current_date", "*_time") is False

    def test_wildcard_contains(self):
        """Test wildcard contains pattern."""
        assert matches_pattern("search_documents", "*search*") is True
        assert matches_pattern("find_documents", "*search*") is False

    def test_full_wildcard(self):
        """Test full wildcard pattern."""
        assert matches_pattern("anything", "*") is True


class TestGetMCPServerNames:
    """Tests for get_mcp_server_names function."""

    def test_empty_servers(self):
        """Test with no MCP servers."""
        config = ModelConfig(name="test", mcp_servers=[])
        assert get_mcp_server_names(config) == []

    def test_list_servers(self):
        """Test with list of servers."""
        config = ModelConfig(name="test", mcp_servers=["server1", "server2"])
        assert get_mcp_server_names(config) == ["server1", "server2"]

    def test_dict_servers(self):
        """Test with dict of servers."""
        config = ModelConfig(
            name="test",
            mcp_servers={
                "server1": {"whitelist": ["tool1"]},
                "server2": None
            }
        )
        names = get_mcp_server_names(config)
        assert "server1" in names
        assert "server2" in names


class TestGetMCPServerFilters:
    """Tests for get_mcp_server_filters function."""

    def test_list_servers_no_filters(self):
        """Test that list servers have no filters."""
        config = ModelConfig(name="test", mcp_servers=["server1"])
        whitelist, blacklist = get_mcp_server_filters(config, "server1")
        
        assert whitelist == []
        assert blacklist == []

    def test_dict_server_with_whitelist(self):
        """Test server with whitelist."""
        config = ModelConfig(
            name="test",
            mcp_servers={
                "server1": {"whitelist": ["tool1", "tool2"]}
            }
        )
        whitelist, blacklist = get_mcp_server_filters(config, "server1")
        
        assert whitelist == ["tool1", "tool2"]
        assert blacklist == []

    def test_dict_server_with_blacklist(self):
        """Test server with blacklist."""
        config = ModelConfig(
            name="test",
            mcp_servers={
                "server1": {"blacklist": ["tool3"]}
            }
        )
        whitelist, blacklist = get_mcp_server_filters(config, "server1")
        
        assert whitelist == []
        assert blacklist == ["tool3"]

    def test_dict_server_with_both(self):
        """Test server with both whitelist and blacklist."""
        config = ModelConfig(
            name="test",
            mcp_servers={
                "server1": {
                    "whitelist": ["tool1", "tool2"],
                    "blacklist": ["tool3"]
                }
            }
        )
        whitelist, blacklist = get_mcp_server_filters(config, "server1")
        
        assert whitelist == ["tool1", "tool2"]
        assert blacklist == ["tool3"]

    def test_dict_server_none_config(self):
        """Test server with None config."""
        config = ModelConfig(
            name="test",
            mcp_servers={"server1": None}
        )
        whitelist, blacklist = get_mcp_server_filters(config, "server1")
        
        assert whitelist == []
        assert blacklist == []

    def test_unknown_server(self):
        """Test unknown server returns empty filters."""
        config = ModelConfig(
            name="test",
            mcp_servers={"server1": {"whitelist": ["tool1"]}}
        )
        whitelist, blacklist = get_mcp_server_filters(config, "unknown")
        
        assert whitelist == []
        assert blacklist == []


class TestGetProxyToolNames:
    """Tests for get_proxy_tool_names function."""

    def test_no_tools(self):
        """Test with no tools."""
        config = ModelConfig(name="test")
        names = get_proxy_tool_names(config, [])
        
        assert names == set()

    def test_model_tools(self):
        """Test with model-defined tools."""
        config = ModelConfig(
            name="test",
            tools=[
                Tool(function=ToolFunction(name="tool1")),
                Tool(function=ToolFunction(name="tool2")),
            ]
        )
        names = get_proxy_tool_names(config, [])
        
        assert names == {"tool1", "tool2"}

    def test_mcp_tools(self):
        """Test with MCP tools."""
        config = ModelConfig(name="test")
        mcp_tools = [
            MCPTool(name="search", server_name="homeassistant", description="Search"),
            MCPTool(name="control", server_name="homeassistant", description="Control"),
        ]
        names = get_proxy_tool_names(config, mcp_tools)
        
        assert "mcp_homeassistant_search" in names
        assert "mcp_homeassistant_control" in names

    def test_mcp_tools_with_whitelist(self):
        """Test MCP tools with whitelist filter."""
        config = ModelConfig(
            name="test",
            mcp_servers={"server": {"whitelist": ["allowed*"]}}
        )
        mcp_tools = [
            MCPTool(name="allowed_tool", server_name="server", description=""),
            MCPTool(name="blocked_tool", server_name="server", description=""),
        ]
        names = get_proxy_tool_names(config, mcp_tools)
        
        assert "mcp_server_allowed_tool" in names
        assert "mcp_server_blocked_tool" not in names

    def test_mcp_tools_with_blacklist(self):
        """Test MCP tools with blacklist filter."""
        config = ModelConfig(
            name="test",
            mcp_servers={"server": {"blacklist": ["blocked*"]}}
        )
        mcp_tools = [
            MCPTool(name="allowed_tool", server_name="server", description=""),
            MCPTool(name="blocked_tool", server_name="server", description=""),
        ]
        names = get_proxy_tool_names(config, mcp_tools)
        
        assert "mcp_server_allowed_tool" in names
        assert "mcp_server_blocked_tool" not in names


class TestMergeTools:
    """Tests for merge_tools function."""

    def test_empty_tools(self):
        """Test merging with no tools."""
        config = ModelConfig(name="test")
        result = merge_tools(None, [], [], config)
        
        assert result == []

    def test_proxy_tools_only(self):
        """Test with only proxy tools."""
        config = ModelConfig(name="test")
        proxy_tools = [
            Tool(function=ToolFunction(name="tool1", description="Tool 1")),
        ]
        result = merge_tools(None, proxy_tools, [], config)
        
        assert len(result) == 1
        assert result[0].function.name == "tool1"

    def test_mcp_tools_only(self):
        """Test with only MCP tools."""
        config = ModelConfig(name="test")
        mcp_tools = [
            MCPTool(name="search", server_name="server", description="Search things"),
        ]
        result = merge_tools(None, [], mcp_tools, config)
        
        assert len(result) == 1
        assert result[0].function.name == "mcp_server_search"

    def test_client_tools_only(self):
        """Test with only client tools."""
        config = ModelConfig(name="test")
        client_tools = [
            ToolSchema(function=ToolFunctionSchema(name="client_tool")),
        ]
        result = merge_tools(client_tools, [], [], config)
        
        assert len(result) == 1
        assert result[0].function.name == "client_tool"

    def test_client_tools_override(self):
        """Test that client tools override proxy tools with same name."""
        config = ModelConfig(name="test")
        proxy_tools = [
            Tool(function=ToolFunction(name="shared_tool", description="Proxy version")),
        ]
        client_tools = [
            ToolSchema(function=ToolFunctionSchema(name="shared_tool", description="Client version")),
        ]
        result = merge_tools(client_tools, proxy_tools, [], config)
        
        assert len(result) == 1
        assert result[0].function.description == "Client version"

    def test_mcp_tools_filtered(self):
        """Test that MCP tools are filtered."""
        config = ModelConfig(
            name="test",
            mcp_servers={"server": {"whitelist": ["allowed"]}}
        )
        mcp_tools = [
            MCPTool(name="allowed", server_name="server", description=""),
            MCPTool(name="blocked", server_name="server", description=""),
        ]
        result = merge_tools(None, [], mcp_tools, config)
        
        names = [t.function.name for t in result]
        assert "mcp_server_allowed" in names
        assert "mcp_server_blocked" not in names


class TestBuildSystemPrompt:
    """Tests for build_system_prompt function."""

    def test_no_system_prompt(self):
        """Test model with no system prompt."""
        config = ModelConfig(name="test")
        
        with patch("openai_proxy.proxy.get_skills_loader") as mock_loader:
            mock_loader.return_value.get_skills_for_model.return_value = []
            result = build_system_prompt(config)
        
        assert result is None

    def test_system_prompt_only(self):
        """Test model with system prompt but no skills."""
        config = ModelConfig(name="test", system_prompt="You are helpful.")
        
        with patch("openai_proxy.proxy.get_skills_loader") as mock_loader:
            mock_loader.return_value.get_skills_for_model.return_value = []
            result = build_system_prompt(config)
        
        assert result == "You are helpful."

    def test_system_prompt_with_skills(self):
        """Test model with system prompt and skills."""
        from openai_proxy.skills import Skill
        
        config = ModelConfig(name="test", system_prompt="You are helpful.")
        mock_skill = Skill(name="coding", content="Write clean code.", description="")
        
        with patch("openai_proxy.proxy.get_skills_loader") as mock_loader:
            mock_loader.return_value.get_skills_for_model.return_value = [mock_skill]
            mock_loader.return_value.format_skills_for_prompt.return_value = "## Skill: coding\n\nWrite clean code."
            result = build_system_prompt(config)
        
        assert "You are helpful." in result
        assert "Skills & Capabilities" in result
        assert "Write clean code." in result


class TestPrependSystemPrompt:
    """Tests for prepend_system_prompt function."""

    def test_no_system_prompt(self):
        """Test with no system prompt."""
        messages = [Message(role="user", content="Hi")]
        result = prepend_system_prompt(messages, None)
        
        assert result == messages

    def test_prepend_to_empty_messages(self):
        """Test prepending to empty messages."""
        result = prepend_system_prompt([], "You are helpful.")
        
        assert len(result) == 1
        assert result[0].role == "system"
        assert result[0].content == "You are helpful."

    def test_prepend_when_no_system_exists(self):
        """Test prepending when no system message exists."""
        messages = [Message(role="user", content="Hi")]
        result = prepend_system_prompt(messages, "You are helpful.")
        
        assert len(result) == 2
        assert result[0].role == "system"
        assert result[0].content == "You are helpful."
        assert result[1].role == "user"

    def test_combine_with_existing_system(self):
        """Test combining with existing system message."""
        messages = [
            Message(role="system", content="Existing prompt."),
            Message(role="user", content="Hi"),
        ]
        result = prepend_system_prompt(messages, "Prepended prompt.")
        
        assert len(result) == 2
        assert result[0].role == "system"
        assert "Prepended prompt." in result[0].content
        assert "Existing prompt." in result[0].content


class TestExecuteProxyTool:
    """Tests for execute_proxy_tool function."""

    @pytest.mark.asyncio
    async def test_execute_builtin_tool(self):
        """Test executing a built-in tool."""
        with patch("openai_proxy.proxy.get_chat_logger") as mock_logger:
            mock_logger.return_value.log_tool_call = MagicMock()
            
            result = await execute_proxy_tool("get_current_time", {})
        
        data = json.loads(result)
        assert "time" in data

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing an unknown tool."""
        with patch("openai_proxy.proxy.get_chat_logger") as mock_logger:
            mock_logger.return_value.log_tool_call = MagicMock()
            
            result = await execute_proxy_tool("unknown_tool", {})
        
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_mcp_tool(self):
        """Test executing an MCP tool."""
        mock_mcp = AsyncMock()
        mock_mcp.call_tool.return_value = {"status": "success"}
        
        with patch("openai_proxy.proxy.get_mcp_client", return_value=mock_mcp):
            with patch("openai_proxy.proxy.get_chat_logger") as mock_logger:
                mock_logger.return_value.log_tool_call = MagicMock()
                
                result = await execute_proxy_tool("mcp_server_tool", {"arg": "value"})
        
        data = json.loads(result)
        assert data["status"] == "success"
        mock_mcp.call_tool.assert_called_once_with("mcp_server_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_execute_tool_logs_call(self):
        """Test that tool execution logs the call."""
        with patch("openai_proxy.proxy.get_chat_logger") as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value.log_tool_call = mock_log
            
            await execute_proxy_tool("get_current_time", {"tz": "UTC"}, "conv-123")
        
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[0][0] == "conv-123"
        assert call_args[0][1] == "get_current_time"


class TestProxyIntegration:
    """Integration tests for proxy logic."""

    @pytest.fixture
    def model_config(self):
        """Create a test model config."""
        return ModelConfig(
            name="test-assistant",
            upstream_model="gpt-4o-mini",
            system_prompt="You are a test assistant.",
            tools=[
                Tool(function=ToolFunction(
                    name="get_current_time",
                    description="Get current time",
                    parameters={"type": "object", "properties": {}, "required": []}
                ))
            ],
        )

    @pytest.fixture
    def chat_request(self):
        """Create a test chat request."""
        return ChatCompletionRequest(
            model="test-assistant",
            messages=[
                Message(role="user", content="What time is it?"),
            ],
        )

    @pytest.mark.asyncio
    async def test_full_proxy_flow_mock(self, model_config, chat_request):
        """Test a full proxy flow with mocked upstream."""
        from openai_proxy.proxy import process_chat_completion
        
        mock_response = ChatCompletionResponse(
            id="test-123",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="The time is 12:00."),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        )
        
        mock_client = AsyncMock()
        mock_client.chat_completion.return_value = mock_response
        
        with patch("openai_proxy.proxy.get_upstream_client", return_value=mock_client):
            with patch("openai_proxy.proxy.get_chat_logger") as mock_logger:
                mock_logger.return_value.log_request = MagicMock()
                mock_logger.return_value.log_response = MagicMock()
                mock_logger.return_value._get_conversation_id.return_value = "test-conv"
                
                with patch("openai_proxy.proxy.get_mcp_tools_for_model", return_value=[]):
                    with patch("openai_proxy.proxy.build_system_prompt_async", return_value="You are a test assistant."):
                        response = await process_chat_completion(chat_request, model_config)
        
        assert response.id == "test-123"
        assert response.model == "test-assistant"  # Model name should be updated
        assert response.choices[0].message.content == "The time is 12:00."
