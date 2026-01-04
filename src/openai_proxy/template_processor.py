"""Jinja2-based template processor for system prompts.

This module provides Jinja2-based templating for system prompts with the
ability to:
- Call tools (MCP and builtin) directly in templates
- Include reusable template files
- Access variables from the variable store
- Use all standard Jinja2 features (filters, conditionals, loops, etc.)

Template Syntax:
    Standard Jinja2 syntax is used:
    - {{ expression }} for output
    - {% statement %} for control flow
    - {# comment #} for comments

Tool Calls:
    Tools can be called using the `tool()` function:
        {{ tool("get_current_time") }}
        {{ tool("get_current_time", timezone="UTC") }}
        {{ tool("mcp_homeassistant_list_entities", domain="light") }}

Template Includes:
    Templates from the templates/ directory can be included:
        {% include "common/header.j2" %}
        {% include "prompts/assistant-base.j2" %}

Variables:
    Variables from variables.yaml are available directly:
        {{ app_name }}
        {{ settings.max_tokens }}
        
Examples:
    system_prompt: |
      {% include "base-assistant.j2" %}
      
      ## Current Context
      
      The current date and time is: {{ tool("get_current_time", timezone="UTC") }}
      App version: {{ version }}
      
      {% if debug_mode %}
      Debug mode is enabled.
      {% endif %}
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import (
    BaseLoader,
    Environment,
    FileSystemLoader,
    TemplateNotFound,
    select_autoescape,
)

from .handlers import get_handler
from .mcp_client import get_mcp_client
from .variable_store import get_variable_store

logger = logging.getLogger(__name__)


class ToolCallExtension:
    """Helper class to enable sync-style tool calls in async context.
    
    Since Jinja2 doesn't natively support async, we collect tool calls
    during template rendering and process them afterward.
    """

    def __init__(self):
        self.pending_calls: list[tuple[str, dict[str, Any], str]] = []
        self._counter = 0

    def create_placeholder(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        """Create a placeholder for a tool call.
        
        Args:
            tool_name: Name of the tool to call
            kwargs: Arguments for the tool
            
        Returns:
            Placeholder string that will be replaced with the result
        """
        self._counter += 1
        placeholder = f"__TOOL_PLACEHOLDER_{self._counter}__"
        self.pending_calls.append((tool_name, kwargs, placeholder))
        return placeholder


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


class TemplateProcessor:
    """Jinja2-based template processor with tool calling support.
    
    This processor creates a Jinja2 environment configured with:
    - FileSystemLoader for the templates directory
    - Custom `tool()` function for calling tools
    - Variables from the variable store
    """

    def __init__(self, templates_dir: Path | str | None = None):
        """Initialize the template processor.
        
        Args:
            templates_dir: Path to the templates directory (optional)
        """
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self._env: Environment | None = None

    def _create_environment(self, tool_extension: ToolCallExtension) -> Environment:
        """Create a Jinja2 environment.
        
        Args:
            tool_extension: Tool extension for handling tool calls
            
        Returns:
            Configured Jinja2 Environment
        """
        # Set up loader
        if self.templates_dir and self.templates_dir.exists():
            loader: BaseLoader | None = FileSystemLoader(str(self.templates_dir))
        else:
            loader = None

        env = Environment(
            loader=loader,
            autoescape=select_autoescape(default=False),
            # Keep whitespace control reasonable for prompt templates
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add tool function
        def tool_func(name: str, **kwargs: Any) -> str:
            """Call a tool and return the result.
            
            Args:
                name: Tool name
                **kwargs: Tool arguments
                
            Returns:
                Placeholder that will be replaced with the result
            """
            return tool_extension.create_placeholder(name, kwargs)

        env.globals["tool"] = tool_func
        
        return env

    async def process(
        self,
        template_string: str,
        extra_variables: dict[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> str:
        """Process a template string.
        
        Args:
            template_string: The template string to process
            extra_variables: Additional variables to make available
            raise_on_error: If True, raise exceptions instead of returning error strings
            
        Returns:
            Processed string with all templates and tool calls resolved
            
        Raises:
            Exception: If raise_on_error is True and template processing fails
        """
        # Get variables from store
        store = await get_variable_store()
        variables = await store.load_all()
        
        # Merge with extra variables
        if extra_variables:
            variables = {**variables, **extra_variables}

        # Create tool extension for this render
        tool_extension = ToolCallExtension()
        
        # Create environment
        env = self._create_environment(tool_extension)
        
        # Compile and render template
        try:
            template = env.from_string(template_string)
            rendered = template.render(**variables)
        except TemplateNotFound as e:
            logger.error(f"Template not found: {e}")
            if raise_on_error:
                raise
            return f"[Template error: {e}]"
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            if raise_on_error:
                raise
            return f"[Template error: {e}]"

        # Process tool calls
        if tool_extension.pending_calls:
            logger.info(f"Processing {len(tool_extension.pending_calls)} tool call(s)")
            
            # Execute all tool calls concurrently
            async def execute_call(
                tool_name: str, kwargs: dict[str, Any], placeholder: str
            ) -> tuple[str, str]:
                try:
                    result = await call_tool(tool_name, kwargs)
                    return (placeholder, result)
                except Exception as e:
                    logger.error(f"Error calling tool '{tool_name}': {e}")
                    return (placeholder, f"[Error: {e}]")

            tasks = [
                execute_call(name, kwargs, placeholder)
                for name, kwargs, placeholder in tool_extension.pending_calls
            ]
            results = await asyncio.gather(*tasks)

            # Replace placeholders with results
            for placeholder, result in results:
                rendered = rendered.replace(placeholder, result)

        return rendered

    async def process_file(
        self,
        template_name: str,
        extra_variables: dict[str, Any] | None = None,
    ) -> str:
        """Process a template file from the templates directory.
        
        Args:
            template_name: Name of the template file (relative to templates dir)
            extra_variables: Additional variables to make available
            
        Returns:
            Processed string
        """
        if not self.templates_dir or not self.templates_dir.exists():
            raise ValueError("Templates directory not configured or does not exist")

        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_name}")

        template_string = template_path.read_text()
        return await self.process(template_string, extra_variables)


# Global template processor instance
_template_processor: TemplateProcessor | None = None


def get_template_processor() -> TemplateProcessor:
    """Get the global template processor instance.
    
    Returns:
        The global TemplateProcessor instance
    """
    global _template_processor
    if _template_processor is None:
        _template_processor = TemplateProcessor()
    return _template_processor


def initialize_template_processor(templates_dir: Path | str | None = None) -> TemplateProcessor:
    """Initialize the global template processor.
    
    Args:
        templates_dir: Path to the templates directory
        
    Returns:
        Initialized TemplateProcessor
    """
    global _template_processor
    _template_processor = TemplateProcessor(templates_dir)
    return _template_processor


async def process_system_prompt(prompt: str | None) -> str | None:
    """Process all templates in a system prompt.
    
    This is the main entry point for template processing, compatible
    with the previous API.
    
    Args:
        prompt: The system prompt potentially containing templates
        
    Returns:
        The processed prompt with templates replaced by results
    """
    if not prompt:
        return prompt

    processor = get_template_processor()
    return await processor.process(prompt)
