"""Template processor for system prompts with tool calls.

This module provides the ability to call tools (MCP or builtin) directly
in the system prompt using a template syntax, replacing the template with
the tool's result at runtime.

Template syntax:
    {{tool_name()}}                    - Call a tool with no arguments
    {{tool_name(arg1="value")}}        - Call with named arguments
    {{tool_name(arg1="value", arg2=123)}}  - Multiple arguments
    {{mcp_server_toolname()}}          - Call an MCP tool

Examples:
    system_prompt: |
      You are a helpful assistant.
      The current time is: {{get_current_time(timezone="UTC")}}
      
      Available entities: {{mcp_homeassistant_list_entities(domain="light")}}

The templates are processed asynchronously before the system prompt is
sent to the LLM.
"""

import asyncio
import json
import logging
import re
from typing import Any

from .handlers import get_handler
from .mcp_client import get_mcp_client

logger = logging.getLogger(__name__)

# Regex pattern to match tool templates
# Matches: {{tool_name()}} or {{tool_name(arg1="value", arg2=123)}}
TEMPLATE_PATTERN = re.compile(
    r'\{\{\s*'           # Opening {{
    r'(\w+)'             # Tool name (capture group 1)
    r'\s*\(\s*'          # Opening parenthesis
    r'([^)]*)'           # Arguments (capture group 2) - everything inside ()
    r'\s*\)\s*'          # Closing parenthesis
    r'\}\}'              # Closing }}
)


def parse_arguments(args_str: str) -> dict[str, Any]:
    """Parse argument string into a dictionary.
    
    Supports:
        arg1="string value"
        arg2=123
        arg3=12.5
        arg4=true
        arg5=false
        arg6=null
    
    Args:
        args_str: String like 'arg1="value", arg2=123'
        
    Returns:
        Dictionary of parsed arguments
    """
    if not args_str.strip():
        return {}
    
    result = {}
    
    # Pattern to match individual key=value pairs
    # Handles: key="string", key=123, key=12.5, key=true, key=false, key=null
    arg_pattern = re.compile(
        r'(\w+)\s*=\s*'  # Key and equals sign
        r'(?:'
        r'"([^"]*)"'     # String value in double quotes
        r"|'([^']*)'"    # String value in single quotes
        r'|(\d+\.\d+)'   # Float value
        r'|(\d+)'        # Integer value
        r'|(true|false)' # Boolean value
        r'|(null)'       # Null value
        r')'
    )
    
    for match in arg_pattern.finditer(args_str):
        key = match.group(1)
        
        if match.group(2) is not None:  # Double-quoted string
            result[key] = match.group(2)
        elif match.group(3) is not None:  # Single-quoted string
            result[key] = match.group(3)
        elif match.group(4) is not None:  # Float
            result[key] = float(match.group(4))
        elif match.group(5) is not None:  # Integer
            result[key] = int(match.group(5))
        elif match.group(6) is not None:  # Boolean
            result[key] = match.group(6).lower() == 'true'
        elif match.group(7) is not None:  # Null
            result[key] = None
    
    return result


async def call_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Call a tool by name and return the result as a string.
    
    Handles both built-in tools and MCP tools.
    
    Args:
        tool_name: Name of the tool to call
        arguments: Arguments to pass to the tool
        
    Returns:
        Tool result as a string
    """
    # Check if this is an MCP tool
    if tool_name.startswith("mcp_"):
        mcp_client = await get_mcp_client()
        result = await mcp_client.call_tool(tool_name, arguments)
        if isinstance(result, dict):
            return json.dumps(result)
        return str(result)
    
    # Check built-in handlers
    handler = get_handler(tool_name)
    if handler is not None:
        try:
            return await handler.execute(arguments)
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return f"[Error calling {tool_name}: {e}]"
    
    logger.warning(f"Unknown tool in template: {tool_name}")
    return f"[Unknown tool: {tool_name}]"


async def process_template(template: str) -> str:
    """Process a single template and return the result.
    
    Args:
        template: The template string like 'tool_name(arg1="value")'
        
    Returns:
        The tool's result as a string
    """
    match = TEMPLATE_PATTERN.match("{{" + template + "}}")
    if not match:
        return f"[Invalid template: {template}]"
    
    tool_name = match.group(1)
    args_str = match.group(2)
    arguments = parse_arguments(args_str)
    
    logger.debug(f"Processing template: {tool_name}({arguments})")
    
    return await call_tool(tool_name, arguments)


async def process_system_prompt(prompt: str | None) -> str | None:
    """Process all templates in a system prompt.
    
    Finds all {{tool_name(args)}} patterns and replaces them with
    the tool's result.
    
    Args:
        prompt: The system prompt potentially containing templates
        
    Returns:
        The processed prompt with templates replaced by results
    """
    if not prompt:
        return prompt
    
    # Find all templates
    matches = list(TEMPLATE_PATTERN.finditer(prompt))
    
    if not matches:
        return prompt
    
    logger.info(f"Processing {len(matches)} template(s) in system prompt")
    
    # Process all templates concurrently
    async def process_match(match: re.Match) -> tuple[re.Match, str]:
        tool_name = match.group(1)
        args_str = match.group(2)
        arguments = parse_arguments(args_str)
        
        logger.debug(f"Template: {tool_name}({arguments})")
        
        try:
            result = await call_tool(tool_name, arguments)
            return (match, result)
        except Exception as e:
            logger.error(f"Error processing template '{tool_name}': {e}")
            return (match, f"[Error: {e}]")
    
    # Run all template processing concurrently
    tasks = [process_match(m) for m in matches]
    results = await asyncio.gather(*tasks)
    
    # Replace templates with results (in reverse order to preserve positions)
    processed = prompt
    for match, result in sorted(results, key=lambda x: x[0].start(), reverse=True):
        processed = processed[:match.start()] + result + processed[match.end():]
    
    return processed


def find_templates(prompt: str | None) -> list[dict[str, Any]]:
    """Find all templates in a prompt without processing them.
    
    Useful for validation or debugging.
    
    Args:
        prompt: The prompt to scan for templates
        
    Returns:
        List of dicts with tool_name and arguments for each template
    """
    if not prompt:
        return []
    
    templates = []
    for match in TEMPLATE_PATTERN.finditer(prompt):
        tool_name = match.group(1)
        args_str = match.group(2)
        arguments = parse_arguments(args_str)
        templates.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "template": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })
    
    return templates
