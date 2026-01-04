"""Proxy logic for handling chat completions with tool augmentation."""

import json
import logging
from typing import Any

from .chat_logger import get_chat_logger
from .client import get_upstream_client
from .handlers import get_handler
from .mcp_client import get_mcp_client, MCPTool
from .models import ModelConfig, Tool
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ToolCall,
    ToolSchema,
    ToolFunctionSchema,
)
from .skills import get_skills_loader

logger = logging.getLogger(__name__)


def get_mcp_server_names(model_config: ModelConfig) -> list[str]:
    """Get list of MCP server names from model config."""
    if not model_config.mcp_servers:
        return []
    if isinstance(model_config.mcp_servers, list):
        return model_config.mcp_servers
    return list(model_config.mcp_servers.keys())


def get_mcp_server_filters(model_config: ModelConfig, server_name: str) -> tuple[list[str], list[str]]:
    """Get whitelist and blacklist for a specific MCP server.
    
    Returns:
        Tuple of (whitelist, blacklist) patterns
    """
    if isinstance(model_config.mcp_servers, list):
        return [], []
    
    server_config = model_config.mcp_servers.get(server_name)
    if not server_config:
        return [], []
    
    whitelist = server_config.get("whitelist", []) or []
    blacklist = server_config.get("blacklist", []) or []
    return whitelist, blacklist


def get_proxy_tool_names(
    model_config: ModelConfig,
    mcp_tools: list[MCPTool],
) -> set[str]:
    """Get the names of tools managed by the proxy for this model (after filtering)."""
    names = {tool.function.name for tool in model_config.tools}
    
    # Add MCP tool names with per-server filtering
    for tool in mcp_tools:
        full_name = f"mcp_{tool.server_name}_{tool.name}"
        whitelist, blacklist = get_mcp_server_filters(model_config, tool.server_name)
        
        # Check if tool passes filters (filter on original tool name, not full_name)
        if whitelist and not any(matches_pattern(tool.name, p) for p in whitelist):
            continue
        if blacklist and any(matches_pattern(tool.name, p) for p in blacklist):
            continue
        
        names.add(full_name)
    
    return names


async def get_mcp_tools_for_model(model_config: ModelConfig) -> list[MCPTool]:
    """Get MCP tools for a model based on configured servers."""
    server_names = get_mcp_server_names(model_config)
    if not server_names:
        return []
    
    mcp_client = await get_mcp_client()
    
    # Ensure servers are connected
    for server_name in server_names:
        if server_name not in mcp_client.get_connected_servers():
            await mcp_client.connect_server(server_name)
    
    return mcp_client.get_tools(server_names)


def matches_pattern(name: str, pattern: str) -> bool:
    """Check if a tool name matches a glob pattern.
    
    Supports:
    - Exact match: "get_current_time"
    - Wildcard suffix: "mcp_homeassistant_*"
    - Wildcard prefix: "*_time"
    - Contains: "*search*"
    """
    import fnmatch
    return fnmatch.fnmatch(name, pattern)


def merge_tools(
    client_tools: list[ToolSchema] | None,
    proxy_tools: list[Tool],
    mcp_tools: list[MCPTool],
    model_config: ModelConfig,
) -> list[ToolSchema]:
    """Merge client-provided tools with proxy-configured tools and MCP tools.
    
    Args:
        client_tools: Tools provided by the client (not filtered)
        proxy_tools: Built-in tools from model config
        mcp_tools: Tools from MCP servers
        model_config: Model config with per-server filtering rules
    """
    merged: dict[str, ToolSchema] = {}

    # Add proxy tools first (no filtering on built-in tools)
    for tool in proxy_tools:
        schema = ToolSchema(
            type="function",
            function=ToolFunctionSchema(
                name=tool.function.name,
                description=tool.function.description,
                parameters=tool.function.parameters,
            ),
        )
        merged[tool.function.name] = schema

    # Add MCP tools with per-server filtering
    for tool in mcp_tools:
        whitelist, blacklist = get_mcp_server_filters(model_config, tool.server_name)
        
        # Check if tool passes filters (filter on original tool name)
        if whitelist and not any(matches_pattern(tool.name, p) for p in whitelist):
            logger.debug(f"Tool '{tool.name}' from '{tool.server_name}' excluded by whitelist")
            continue
        if blacklist and any(matches_pattern(tool.name, p) for p in blacklist):
            logger.debug(f"Tool '{tool.name}' from '{tool.server_name}' excluded by blacklist")
            continue
        
        full_name = f"mcp_{tool.server_name}_{tool.name}"
        schema = ToolSchema(
            type="function",
            function=ToolFunctionSchema(
                name=full_name,
                description=tool.description,
                parameters=tool.input_schema or {"type": "object", "properties": {}},
            ),
        )
        merged[full_name] = schema

    # Add/override with client tools (client tools are NOT filtered)
    if client_tools:
        for tool in client_tools:
            merged[tool.function.name] = tool

    return list(merged.values())


def build_system_prompt(model_config: ModelConfig) -> str | None:
    """Build the complete system prompt including skills."""
    parts: list[str] = []

    # Add configured system prompt
    if model_config.system_prompt:
        parts.append(model_config.system_prompt)

    # Add skills
    skills_loader = get_skills_loader()
    skills = skills_loader.get_skills_for_model(
        model_name=model_config.name,
        skill_names=model_config.skills if model_config.skills else None,
        include_global=model_config.include_global_skills,
    )

    if skills:
        skills_content = skills_loader.format_skills_for_prompt(skills)
        if skills_content:
            parts.append("\n\n# Skills & Capabilities\n\n" + skills_content)

    return "\n\n".join(parts) if parts else None


def prepend_system_prompt(
    messages: list[Message],
    system_prompt: str | None,
) -> list[Message]:
    """Prepend a system prompt to the messages if configured."""
    if not system_prompt:
        return messages

    # Check if there's already a system message
    if messages and messages[0].role == "system":
        # Combine with existing system message
        existing_content = messages[0].content or ""
        combined = f"{system_prompt}\n\n{existing_content}"
        return [Message(role="system", content=combined)] + messages[1:]
    else:
        # Prepend new system message
        return [Message(role="system", content=system_prompt)] + messages


async def execute_proxy_tool(
    tool_name: str,
    arguments: dict[str, Any],
    conversation_id: str | None = None,
) -> str:
    """Execute a proxy-managed tool and return the result."""
    chat_logger = get_chat_logger()
    
    # Check if this is an MCP tool
    if tool_name.startswith("mcp_"):
        mcp_client = await get_mcp_client()
        result = await mcp_client.call_tool(tool_name, arguments)
        if isinstance(result, dict):
            result_str = json.dumps(result)
        else:
            result_str = str(result)
        
        # Log the tool call
        if conversation_id:
            chat_logger.log_tool_call(conversation_id, tool_name, arguments, result_str)
        
        return result_str

    # Check built-in handlers
    handler = get_handler(tool_name)
    if handler is None:
        return json.dumps({"error": f"No handler found for tool: {tool_name}"})

    try:
        result_str = await handler.execute(arguments)
        
        # Log the tool call
        if conversation_id:
            chat_logger.log_tool_call(conversation_id, tool_name, arguments, result_str)
        
        return result_str
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


async def handle_tool_calls(
    response: ChatCompletionResponse,
    model_config: ModelConfig,
    original_request: ChatCompletionRequest,
    mcp_tools: list[MCPTool],
    conversation_id: str | None = None,
) -> ChatCompletionResponse:
    """Handle tool calls in the response, executing proxy tools and continuing the conversation."""
    if not response.choices:
        return response

    choice = response.choices[0]
    if not choice.message.tool_calls:
        return response

    proxy_tool_names = get_proxy_tool_names(model_config, mcp_tools)

    # Separate proxy tool calls from client tool calls
    proxy_calls: list[ToolCall] = []
    client_calls: list[ToolCall] = []

    for tool_call in choice.message.tool_calls:
        if tool_call.function.name in proxy_tool_names:
            proxy_calls.append(tool_call)
        else:
            client_calls.append(tool_call)

    # If no proxy tools to handle, return as-is
    if not proxy_calls:
        return response

    # Execute proxy tools
    tool_results: list[Message] = []
    for tool_call in proxy_calls:
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {}

        result = await execute_proxy_tool(tool_call.function.name, arguments, conversation_id)
        tool_results.append(
            Message(
                role="tool",
                content=result,
                tool_call_id=tool_call.id,
            )
        )

    # Build new messages with the assistant's response and tool results
    new_messages = list(original_request.messages)
    new_messages.append(choice.message)
    new_messages.extend(tool_results)

    # Create a new request with updated messages
    new_request = ChatCompletionRequest(
        model=original_request.model,
        messages=new_messages,
        tools=original_request.tools,
        tool_choice=original_request.tool_choice,
        temperature=original_request.temperature,
        top_p=original_request.top_p,
        n=original_request.n,
        stream=False,
        stop=original_request.stop,
        max_tokens=original_request.max_tokens,
        presence_penalty=original_request.presence_penalty,
        frequency_penalty=original_request.frequency_penalty,
    )

    # Continue the conversation
    return await process_chat_completion(new_request, model_config, conversation_id)


async def process_chat_completion(
    request: ChatCompletionRequest,
    model_config: ModelConfig,
    conversation_id: str | None = None,
) -> ChatCompletionResponse:
    """Process a chat completion request with tool augmentation."""
    client = get_upstream_client()
    chat_logger = get_chat_logger()

    # Generate conversation ID if not provided
    if conversation_id is None:
        conversation_id = chat_logger._get_conversation_id(request.model_dump())
    
    logger.debug(f"Processing chat completion for conversation: {conversation_id}")

    # Get MCP tools for this model
    mcp_tools = await get_mcp_tools_for_model(model_config)

    # Build complete system prompt (including skills)
    system_prompt = build_system_prompt(model_config)

    # Prepare the request for upstream (with per-server tool filtering)
    merged_tools = merge_tools(request.tools, model_config.tools, mcp_tools, model_config)
    modified_messages = prepend_system_prompt(request.messages, system_prompt)
    
    modified_request = ChatCompletionRequest(
        model=model_config.effective_upstream_model,
        messages=modified_messages,
        tools=merged_tools or None,
        tool_choice=request.tool_choice,
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        stream=False,
        stop=request.stop,
        max_tokens=request.max_tokens,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        logit_bias=request.logit_bias,
        user=request.user,
    )

    # Log the request
    chat_logger.log_request(
        conversation_id,
        request.model_dump(),
        modified_request.model_dump(),
    )

    # Send to upstream
    response = await client.chat_completion(modified_request)

    # Log the response
    chat_logger.log_response(conversation_id, response.model_dump())

    # Handle any proxy tool calls
    response = await handle_tool_calls(response, model_config, request, mcp_tools, conversation_id)

    # Update the model name in the response to match the requested model
    response.model = request.model

    return response


async def process_chat_completion_stream(
    request: ChatCompletionRequest,
    model_config: ModelConfig,
):
    """Process a streaming chat completion request with tool status updates.
    
    When proxy tools are called, streams status updates so the client can see
    what's happening (similar to "thinking" indicators).
    
    Status events have the format:
    {
        "proxy_status": {
            "type": "tool_execution",
            "status": "thinking|calling_llm|executing_tool|tool_completed|processing_results",
            "tool_name": "optional tool name"
        }
    }
    """
    import time
    import uuid
    from .client import get_upstream_client
    from .schemas import ChatCompletionChunk, StreamChoice, DeltaMessage

    chat_logger = get_chat_logger()
    conversation_id = chat_logger._get_conversation_id(request.model_dump())

    # Get MCP tools for this model
    mcp_tools = await get_mcp_tools_for_model(model_config)

    # Helper to create status chunks
    def make_status_chunk(status: str, tool_name: str | None = None, chunk_id: str | None = None) -> str:
        """Create a status chunk for tool execution progress."""
        chunk_id = chunk_id or f"chatcmpl-{uuid.uuid4().hex[:8]}"
        # Use a custom format that clients can recognize
        chunk = ChatCompletionChunk(
            id=chunk_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        role="assistant",
                        content=None,
                    ),
                    finish_reason=None,
                )
            ],
        )
        # Add custom status field
        data = chunk.model_dump()
        data["proxy_status"] = {
            "type": "tool_execution",
            "status": status,
            "tool_name": tool_name,
        }
        return f"data: {json.dumps(data)}\n\n"

    # --- Handle tools with streaming status ---
    if model_config.tools or mcp_tools:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        # Prepare request for upstream (with per-server tool filtering)
        system_prompt = build_system_prompt(model_config)
        merged_tools = merge_tools(request.tools, model_config.tools, mcp_tools, model_config)
        modified_messages = prepend_system_prompt(request.messages, system_prompt)
        proxy_tool_names = get_proxy_tool_names(model_config, mcp_tools)
        
        current_messages = list(modified_messages)
        iteration = 0
        max_iterations = 10  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            
            modified_request = ChatCompletionRequest(
                model=model_config.effective_upstream_model,
                messages=current_messages,
                tools=merged_tools or None,
                tool_choice=request.tool_choice,
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                stream=False,  # Non-streaming for tool handling
                stop=request.stop,
                max_tokens=request.max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                logit_bias=request.logit_bias,
                user=request.user,
            )
            
            # Log the request on first iteration
            if iteration == 1:
                chat_logger.log_request(
                    conversation_id,
                    request.model_dump(),
                    modified_request.model_dump(),
                )
            
            # Stream status: calling LLM
            yield make_status_chunk("calling_llm", None, chunk_id)
            
            client = get_upstream_client()
            response = await client.chat_completion(modified_request)
            
            if not response.choices:
                break
                
            choice = response.choices[0]
            
            # Check for tool calls
            if not choice.message.tool_calls:
                # No tool calls - stream the final response
                final_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=response.created,
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                role=choice.message.role,
                                content=choice.message.content,
                                tool_calls=choice.message.tool_calls,
                            ),
                            finish_reason=choice.finish_reason,
                        )
                    ],
                )
                chat_logger.log_response(conversation_id, response.model_dump())
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Separate proxy tool calls from client tool calls
            proxy_calls: list[ToolCall] = []
            client_calls: list[ToolCall] = []
            
            for tool_call in choice.message.tool_calls:
                if tool_call.function.name in proxy_tool_names:
                    proxy_calls.append(tool_call)
                else:
                    client_calls.append(tool_call)
            
            # If there are client tool calls, return them to the client
            if client_calls and not proxy_calls:
                final_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=response.created,
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                role=choice.message.role,
                                content=choice.message.content,
                                tool_calls=client_calls,
                            ),
                            finish_reason=choice.finish_reason,
                        )
                    ],
                )
                chat_logger.log_response(conversation_id, response.model_dump())
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Execute proxy tools with streaming status
            tool_results: list[Message] = []
            for tool_call in proxy_calls:
                tool_name = tool_call.function.name
                
                # Stream status: executing tool
                yield make_status_chunk("executing_tool", tool_name, chunk_id)
                
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                result = await execute_proxy_tool(tool_name, arguments, conversation_id)
                tool_results.append(
                    Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                    )
                )
                
                # Stream status: tool completed
                yield make_status_chunk("tool_completed", tool_name, chunk_id)
            
            # Add assistant message and tool results to conversation
            current_messages.append(choice.message)
            current_messages.extend(tool_results)
            
            # If there were also client tool calls, we need to handle them
            if client_calls:
                # Return both the tool results we got and the client calls
                final_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=response.created,
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(
                                role=choice.message.role,
                                content=choice.message.content,
                                tool_calls=client_calls,
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
                chat_logger.log_response(conversation_id, response.model_dump())
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Continue loop to let LLM process tool results
            yield make_status_chunk("processing_results", None, chunk_id)
        
        # Max iterations reached
        logger.warning(f"Max tool iterations ({max_iterations}) reached for conversation {conversation_id}")
        yield "data: [DONE]\n\n"
        return

    # No proxy tools - stream directly from upstream
    client = get_upstream_client()

    # Build complete system prompt (including skills)
    system_prompt = build_system_prompt(model_config)
    modified_messages = prepend_system_prompt(request.messages, system_prompt)

    modified_request = ChatCompletionRequest(
        model=model_config.effective_upstream_model,
        messages=modified_messages,
        tools=request.tools,  # Pass through client tools only
        tool_choice=request.tool_choice,
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        stream=True,
        stop=request.stop,
        max_tokens=request.max_tokens,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        logit_bias=request.logit_bias,
        user=request.user,
    )

    # Log the request
    chat_logger.log_request(
        conversation_id,
        request.model_dump(),
        modified_request.model_dump(),
    )

    # Collect streamed content for logging
    collected_content = []
    collected_response: dict | None = None

    async for chunk in client.chat_completion_stream(modified_request):
        if chunk == "[DONE]":
            # Log the collected response
            if collected_response:
                collected_response["choices"] = [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(collected_content)},
                    "finish_reason": "stop",
                }]
                chat_logger.log_response(conversation_id, collected_response)
            yield "data: [DONE]\n\n"
        else:
            # Parse and modify the model name in the chunk
            try:
                data = json.loads(chunk)
                data["model"] = request.model
                
                # Collect content for logging
                if collected_response is None:
                    collected_response = {"id": data.get("id"), "model": data.get("model")}
                if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                    collected_content.append(data["choices"][0]["delta"]["content"])
                
                yield f"data: {json.dumps(data)}\n\n"
            except json.JSONDecodeError:
                yield f"data: {chunk}\n\n"
