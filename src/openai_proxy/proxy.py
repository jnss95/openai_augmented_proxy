"""Proxy logic for handling chat completions with tool augmentation."""

import json
import logging
from typing import Any

from .chat_logger import get_chat_logger
from .client import get_upstream_client
from .experts import (
    EXPERT_TOOL_NAME,
    build_experts_system_prompt_section,
    execute_expert_call,
    get_expert_tool_schema,
    prepare_history_for_expert,
)
from .handlers import get_handler
from .mcp_client import get_mcp_client, MCPTool
from .models import ModelConfig, Tool, get_model_registry
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ToolCall,
    ToolSchema,
    ToolFunctionSchema,
)
from .skills import get_skills_loader
from .template_processor import process_system_prompt

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
    
    # Add expert tool if experts are configured
    if model_config.experts:
        names.add(EXPERT_TOOL_NAME)
    
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

    # Add expert tool if experts are configured
    if model_config.experts:
        expert_schema = get_expert_tool_schema()
        merged[EXPERT_TOOL_NAME] = ToolSchema(
            type="function",
            function=ToolFunctionSchema(
                name=expert_schema["function"]["name"],
                description=expert_schema["function"]["description"],
                parameters=expert_schema["function"]["parameters"],
            ),
        )

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
    """Build the complete system prompt including skills (without template processing).
    
    Note: This returns the raw prompt. Use build_system_prompt_async() to also
    process templates with tool calls.
    """
    parts: list[str] = []

    # Add configured system prompt
    if model_config.system_prompt:
        parts.append(model_config.system_prompt)

    # Add experts section if experts are configured
    if model_config.experts:
        model_registry = get_model_registry()
        experts_section = build_experts_system_prompt_section(
            model_config.experts,
            model_registry,
        )
        if experts_section:
            parts.append(experts_section)

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


async def build_system_prompt_async(model_config: ModelConfig) -> str | None:
    """Build the complete system prompt including skills and process Jinja2 templates.
    
    This function builds the system prompt and processes Jinja2 templates including
    tool calls like {{ tool("name", arg="value") }}, includes, and variables.
    """
    raw_prompt = build_system_prompt(model_config)
    
    if raw_prompt:
        # Process any templates in the prompt
        return await process_system_prompt(raw_prompt)
    
    return raw_prompt


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
    model_config: ModelConfig | None = None,
    original_messages: list[Message] | None = None,
) -> str:
    """Execute a proxy-managed tool and return the result."""
    chat_logger = get_chat_logger()
    
    # Check if this is an expert call
    if tool_name == EXPERT_TOOL_NAME:
        expert_name = arguments.get("expert_model", "")
        prompt = arguments.get("prompt", "")
        chat_history = arguments.get("chat_history")
        
        # If chat_history not provided in arguments but we have original messages,
        # prepare history based on the expert's configured history mode
        if chat_history is None and model_config and original_messages:
            expert_config = model_config.experts.get(expert_name)
            if expert_config:
                chat_history = prepare_history_for_expert(
                    original_messages,
                    expert_config.history,
                )
        
        model_registry = get_model_registry()
        result_str = await execute_expert_call(
            expert_name=expert_name,
            prompt=prompt,
            chat_history=chat_history,
            model_registry=model_registry,
            process_chat_completion_func=process_chat_completion,
        )
        
        # Log the tool call
        if conversation_id:
            chat_logger.log_tool_call(conversation_id, tool_name, arguments, result_str)
        
        return result_str
    
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

        result = await execute_proxy_tool(
            tool_call.function.name,
            arguments,
            conversation_id,
            model_config=model_config,
            original_messages=original_request.messages,
        )
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

    # Build complete system prompt (including skills and template processing)
    system_prompt = await build_system_prompt_async(model_config)

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
