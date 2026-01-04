"""Tests for experts.py - Expert model delegation support."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai_proxy.experts import (
    EXPERT_TOOL_NAME,
    build_experts_system_prompt_section,
    execute_expert_call,
    get_expert_tool_schema,
    prepare_history_for_expert,
)
from openai_proxy.models import ExpertConfig, ModelConfig
from openai_proxy.schemas import Message


class TestGetExpertToolSchema:
    """Tests for get_expert_tool_schema function."""

    def test_returns_valid_schema(self):
        """Test that the schema has all required fields."""
        schema = get_expert_tool_schema()
        
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == EXPERT_TOOL_NAME
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_parameters_structure(self):
        """Test that parameters have correct structure."""
        schema = get_expert_tool_schema()
        params = schema["function"]["parameters"]
        
        assert params["type"] == "object"
        assert "expert_model" in params["properties"]
        assert "prompt" in params["properties"]
        assert "chat_history" in params["properties"]
        assert "expert_model" in params["required"]
        assert "prompt" in params["required"]

    def test_chat_history_is_optional(self):
        """Test that chat_history is not required."""
        schema = get_expert_tool_schema()
        params = schema["function"]["parameters"]
        
        assert "chat_history" not in params["required"]


class TestBuildExpertsSystemPromptSection:
    """Tests for build_experts_system_prompt_section function."""

    def test_empty_experts(self):
        """Test with no experts."""
        result = build_experts_system_prompt_section({}, MagicMock())
        assert result == ""

    def test_single_expert_with_description(self):
        """Test with one expert that has a description."""
        mock_registry = MagicMock()
        mock_expert_model = MagicMock()
        mock_expert_model.description = "A research expert"
        mock_registry.get.return_value = mock_expert_model
        
        experts = {"augmented/researcher": ExpertConfig(history="full")}
        result = build_experts_system_prompt_section(experts, mock_registry)
        
        assert "augmented/researcher" in result
        assert "A research expert" in result
        assert "Full chat history will be provided" in result
        assert "call_expert" in result

    def test_multiple_experts(self):
        """Test with multiple experts."""
        mock_registry = MagicMock()
        
        def mock_get(name):
            models = {
                "expert1": MagicMock(description="Expert one"),
                "expert2": MagicMock(description="Expert two"),
            }
            return models.get(name)
        
        mock_registry.get.side_effect = mock_get
        
        experts = {
            "expert1": ExpertConfig(history="full"),
            "expert2": ExpertConfig(history="condensed"),
        }
        result = build_experts_system_prompt_section(experts, mock_registry)
        
        assert "expert1" in result
        assert "expert2" in result
        assert "Expert one" in result
        assert "Expert two" in result

    def test_expert_without_description(self):
        """Test with expert that has no description."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock(description=None)
        
        experts = {"expert": ExpertConfig(history="off")}
        result = build_experts_system_prompt_section(experts, mock_registry)
        
        assert "No description available" in result

    def test_unknown_expert(self):
        """Test with expert not found in registry."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        
        experts = {"unknown_expert": ExpertConfig(history="full")}
        result = build_experts_system_prompt_section(experts, mock_registry)
        
        assert "unknown_expert" in result
        assert "No description available" in result

    def test_history_modes_in_prompt(self):
        """Test that different history modes are described correctly."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock(description="Test")
        
        # Test full
        result = build_experts_system_prompt_section(
            {"e": ExpertConfig(history="full")}, mock_registry
        )
        assert "Full chat history will be provided" in result
        
        # Test condensed
        result = build_experts_system_prompt_section(
            {"e": ExpertConfig(history="condensed")}, mock_registry
        )
        assert "condensed summary" in result
        
        # Test off
        result = build_experts_system_prompt_section(
            {"e": ExpertConfig(history="off")}, mock_registry
        )
        assert "No chat history will be provided" in result


class TestPrepareHistoryForExpert:
    """Tests for prepare_history_for_expert function."""

    def test_history_mode_off(self):
        """Test that 'off' mode returns empty list."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        result = prepare_history_for_expert(messages, "off")
        assert result == []

    def test_history_mode_full(self):
        """Test that 'full' mode returns all user/assistant messages."""
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        result = prepare_history_for_expert(messages, "full")
        
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there!"}
        assert result[2] == {"role": "user", "content": "How are you?"}

    def test_history_mode_full_filters_system_and_tool(self):
        """Test that full mode excludes system and tool messages."""
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
            Message(role="tool", content='{"result": "data"}', tool_call_id="123"),
            Message(role="assistant", content="Got it"),
        ]
        result = prepare_history_for_expert(messages, "full")
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_history_mode_condensed_few_messages(self):
        """Test condensed mode with few messages returns all."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]
        result = prepare_history_for_expert(messages, "condensed")
        
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_history_mode_condensed_many_messages(self):
        """Test condensed mode with many messages summarizes older ones."""
        messages = [
            Message(role="user", content=f"Message {i}") 
            for i in range(10)
        ]
        # Add assistant responses
        full_messages = []
        for i, msg in enumerate(messages):
            full_messages.append(msg)
            full_messages.append(Message(role="assistant", content=f"Response {i}"))
        
        result = prepare_history_for_expert(full_messages, "condensed")
        
        # Should have summary + recent messages
        assert len(result) > 0
        # First should be a summary (as assistant role)
        assert result[0]["role"] == "assistant"
        assert "[Earlier conversation summary]" in result[0]["content"]

    def test_history_mode_unknown(self):
        """Test unknown mode returns empty list."""
        messages = [Message(role="user", content="Hello")]
        result = prepare_history_for_expert(messages, "unknown_mode")
        assert result == []

    def test_messages_without_content_filtered(self):
        """Test that messages without content are filtered."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content=None),
            Message(role="user", content=""),
            Message(role="assistant", content="Response"),
        ]
        result = prepare_history_for_expert(messages, "full")
        
        # Only messages with content
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Response"


class TestExecuteExpertCall:
    """Tests for execute_expert_call function."""

    @pytest.mark.asyncio
    async def test_expert_not_found(self):
        """Test error when expert model not found."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_model1 = MagicMock()
        mock_model1.name = "model1"
        mock_registry.list_models.return_value = [mock_model1]
        
        result = await execute_expert_call(
            expert_name="unknown_expert",
            prompt="Hello",
            chat_history=None,
            model_registry=mock_registry,
            process_chat_completion_func=AsyncMock(),
        )
        
        result_data = json.loads(result)
        assert "error" in result_data
        assert "unknown_expert" in result_data["error"]

    @pytest.mark.asyncio
    async def test_successful_expert_call(self):
        """Test successful call to expert."""
        mock_registry = MagicMock()
        mock_expert_config = ModelConfig(name="expert1")
        mock_registry.get.return_value = mock_expert_config
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Expert response"
        
        mock_process = AsyncMock(return_value=mock_response)
        
        result = await execute_expert_call(
            expert_name="expert1",
            prompt="What is the weather?",
            chat_history=[{"role": "user", "content": "Previous question"}],
            model_registry=mock_registry,
            process_chat_completion_func=mock_process,
        )
        
        assert result == "Expert response"
        mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_expert_call_with_history(self):
        """Test that chat history is included in request."""
        mock_registry = MagicMock()
        mock_expert_config = ModelConfig(name="expert1")
        mock_registry.get.return_value = mock_expert_config
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        
        mock_process = AsyncMock(return_value=mock_response)
        
        await execute_expert_call(
            expert_name="expert1",
            prompt="Current question",
            chat_history=[
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
            ],
            model_registry=mock_registry,
            process_chat_completion_func=mock_process,
        )
        
        # Verify the request was built with history
        call_args = mock_process.call_args
        request = call_args[0][0]
        
        # Should have: 2 history messages + 1 prompt message
        assert len(request.messages) == 3
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "First message"
        assert request.messages[2].content == "Current question"

    @pytest.mark.asyncio
    async def test_expert_empty_response(self):
        """Test handling of empty expert response."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = ModelConfig(name="expert1")
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        
        mock_process = AsyncMock(return_value=mock_response)
        
        result = await execute_expert_call(
            expert_name="expert1",
            prompt="Hello",
            chat_history=None,
            model_registry=mock_registry,
            process_chat_completion_func=mock_process,
        )
        
        result_data = json.loads(result)
        assert "error" in result_data
        assert "empty" in result_data["error"].lower()

    @pytest.mark.asyncio
    async def test_expert_call_exception(self):
        """Test handling of exceptions during expert call."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = ModelConfig(name="expert1")
        
        mock_process = AsyncMock(side_effect=Exception("Connection error"))
        
        result = await execute_expert_call(
            expert_name="expert1",
            prompt="Hello",
            chat_history=None,
            model_registry=mock_registry,
            process_chat_completion_func=mock_process,
        )
        
        result_data = json.loads(result)
        assert "error" in result_data
        assert "Connection error" in result_data["error"]


class TestExpertToolName:
    """Tests for the EXPERT_TOOL_NAME constant."""

    def test_tool_name_value(self):
        """Test that the tool name is as expected."""
        assert EXPERT_TOOL_NAME == "call_expert"
