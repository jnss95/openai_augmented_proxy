"""OpenAI-compatible API schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field


# Chat completion request/response schemas
class FunctionCall(BaseModel):
    """Function call in a message."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call in a message."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    """Chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ToolFunctionSchema(BaseModel):
    """Tool function schema for requests."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolSchema(BaseModel):
    """Tool schema for requests."""

    type: Literal["function"] = "function"
    function: ToolFunctionSchema


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""

    model: str
    messages: list[Message]
    tools: list[ToolSchema] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None

    # Additional fields that some APIs support
    model_config = {"extra": "allow"}


class Choice(BaseModel):
    """Choice in a chat completion response."""

    index: int
    message: Message
    finish_reason: str | None = None


class Usage(BaseModel):
    """Usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None


class DeltaMessage(BaseModel):
    """Delta message for streaming."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class StreamChoice(BaseModel):
    """Choice in a streaming response."""

    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """Streaming chat completion chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


# Models endpoint schemas
class ModelObject(BaseModel):
    """Model object for /v1/models endpoint."""

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """Response for /v1/models endpoint."""

    object: str = "list"
    data: list[ModelObject]
