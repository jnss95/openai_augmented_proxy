"""Tests for schemas.py - OpenAI-compatible API schemas."""

import pytest
from pydantic import ValidationError

from openai_proxy.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    DeltaMessage,
    FunctionCall,
    Message,
    ModelListResponse,
    ModelObject,
    StreamChoice,
    ToolCall,
    ToolFunctionSchema,
    ToolSchema,
    Usage,
)


class TestMessage:
    """Tests for Message schema."""

    def test_system_message(self):
        """Test creating a system message."""
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."
        assert msg.tool_calls is None

    def test_user_message(self):
        """Test creating a user message."""
        msg = Message(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_assistant_message_with_content(self):
        """Test creating an assistant message with content."""
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_assistant_message_with_tool_calls(self):
        """Test creating an assistant message with tool calls."""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="get_time", arguments='{"tz": "UTC"}'),
        )
        msg = Message(role="assistant", content=None, tool_calls=[tool_call])
        
        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_time"

    def test_tool_message(self):
        """Test creating a tool response message."""
        msg = Message(role="tool", content='{"time": "12:00"}', tool_call_id="call_123")
        assert msg.role == "tool"
        assert msg.content == '{"time": "12:00"}'
        assert msg.tool_call_id == "call_123"

    def test_invalid_role(self):
        """Test that invalid role raises validation error."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="test")


class TestToolCall:
    """Tests for ToolCall schema."""

    def test_function_tool_call(self):
        """Test creating a function tool call."""
        tool_call = ToolCall(
            id="call_abc",
            type="function",
            function=FunctionCall(name="calculator", arguments='{"expr": "2+2"}'),
        )
        assert tool_call.id == "call_abc"
        assert tool_call.type == "function"
        assert tool_call.function.name == "calculator"
        assert tool_call.function.arguments == '{"expr": "2+2"}'

    def test_default_type(self):
        """Test that type defaults to 'function'."""
        tool_call = ToolCall(
            id="call_123",
            function=FunctionCall(name="test", arguments="{}"),
        )
        assert tool_call.type == "function"


class TestToolSchema:
    """Tests for ToolSchema."""

    def test_tool_schema_creation(self):
        """Test creating a tool schema."""
        schema = ToolSchema(
            type="function",
            function=ToolFunctionSchema(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            )
        )
        assert schema.type == "function"
        assert schema.function.name == "get_weather"
        assert schema.function.description == "Get weather for a location"
        assert "location" in schema.function.parameters["properties"]

    def test_tool_schema_default_type(self):
        """Test that tool schema defaults type to function."""
        schema = ToolSchema(
            function=ToolFunctionSchema(name="test")
        )
        assert schema.type == "function"


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest schema."""

    def test_minimal_request(self):
        """Test creating a minimal request."""
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
        )
        assert request.model == "gpt-4"
        assert len(request.messages) == 1
        assert request.temperature is None
        assert request.stream is None

    def test_full_request(self):
        """Test creating a full request with all options."""
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello!"),
            ],
            tools=[
                ToolSchema(
                    function=ToolFunctionSchema(
                        name="test_tool",
                        description="A test tool"
                    )
                )
            ],
            temperature=0.7,
            top_p=0.9,
            n=1,
            stream=True,
            max_tokens=100,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            user="test-user",
        )
        
        assert request.model == "gpt-4"
        assert len(request.messages) == 2
        assert len(request.tools) == 1
        assert request.temperature == 0.7
        assert request.stream is True
        assert request.max_tokens == 100

    def test_request_extra_fields_allowed(self):
        """Test that extra fields are allowed (for API compatibility)."""
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hi")],
            custom_field="custom_value",
        )
        assert request.model == "gpt-4"

    def test_model_dump_excludes_none(self):
        """Test that model_dump with exclude_none removes None values."""
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hi")],
        )
        data = request.model_dump(exclude_none=True)
        
        assert "model" in data
        assert "messages" in data
        assert "temperature" not in data
        assert "stream" not in data


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse schema."""

    def test_response_creation(self):
        """Test creating a chat completion response."""
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1700000000,
            model="gpt-4",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=5,
                completion_tokens=3,
                total_tokens=8,
            ),
        )
        
        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion"
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello!"
        assert response.usage.total_tokens == 8

    def test_response_with_tool_calls(self):
        """Test response with tool calls in the message."""
        response = ChatCompletionResponse(
            id="chatcmpl-456",
            created=1700000000,
            model="gpt-4",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_123",
                                function=FunctionCall(
                                    name="get_time",
                                    arguments="{}"
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
        
        assert response.choices[0].finish_reason == "tool_calls"
        assert len(response.choices[0].message.tool_calls) == 1


class TestChatCompletionChunk:
    """Tests for streaming ChatCompletionChunk schema."""

    def test_chunk_creation(self):
        """Test creating a streaming chunk."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1700000000,
            model="gpt-4",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        
        assert chunk.id == "chatcmpl-123"
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "Hello"

    def test_final_chunk(self):
        """Test creating a final chunk with finish_reason."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            created=1700000000,
            model="gpt-4",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=None),
                    finish_reason="stop",
                )
            ],
        )
        
        assert chunk.choices[0].finish_reason == "stop"


class TestModelObject:
    """Tests for ModelObject schema."""

    def test_model_object_creation(self):
        """Test creating a model object."""
        model = ModelObject(
            id="gpt-4",
            created=1700000000,
            owned_by="openai",
        )
        
        assert model.id == "gpt-4"
        assert model.object == "model"
        assert model.created == 1700000000
        assert model.owned_by == "openai"

    def test_model_list_response(self):
        """Test creating a model list response."""
        response = ModelListResponse(
            data=[
                ModelObject(id="gpt-4", created=1700000000, owned_by="openai"),
                ModelObject(id="gpt-3.5-turbo", created=1699000000, owned_by="openai"),
            ]
        )
        
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].id == "gpt-4"
