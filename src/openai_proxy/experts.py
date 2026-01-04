"""Expert model delegation support.

This module handles the ability for models to call other models (experts).
When a model has experts configured, it gets:
1. A tool to call experts
2. A system prompt section describing available experts
"""

import json
import logging
from typing import TYPE_CHECKING, Any

from .schemas import Message

if TYPE_CHECKING:
    from .models import ExpertConfig, ModelConfig

logger = logging.getLogger(__name__)


# Tool name for expert calls
EXPERT_TOOL_NAME = "call_expert"


def get_expert_tool_schema() -> dict[str, Any]:
    """Get the JSON schema for the expert tool."""
    return {
        "type": "function",
        "function": {
            "name": EXPERT_TOOL_NAME,
            "description": (
                "Delegate a query to a specialized expert model. Use this when a user's request "
                "requires specialized knowledge or capabilities that an expert model can better handle."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expert_model": {
                        "type": "string",
                        "description": "The name of the expert model to delegate to.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The prompt to send to the expert. This should be a clear, well-formulated "
                            "request that contains all necessary context for the expert to respond."
                        ),
                    },
                    "chat_history": {
                        "type": "array",
                        "description": (
                            "The structured chat history to provide context to the expert. "
                            "Format: [{\"role\": \"user\"|\"assistant\", \"content\": \"...\"}]"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["user", "assistant"],
                                },
                                "content": {
                                    "type": "string",
                                },
                            },
                            "required": ["role", "content"],
                        },
                    },
                },
                "required": ["expert_model", "prompt"],
            },
        },
    }


def build_experts_system_prompt_section(
    experts: dict[str, "ExpertConfig"],
    model_registry: "Any",
) -> str:
    """Build the system prompt section describing available experts.
    
    Args:
        experts: Dict of expert model names to their config
        model_registry: The model registry to look up expert descriptions
        
    Returns:
        A formatted string to add to the system prompt
    """
    if not experts:
        return ""
    
    lines = [
        "# Available Expert Models",
        "",
        "You have access to specialized expert models that you can delegate queries to.",
        "Use the `call_expert` tool to route user requests to the appropriate expert.",
        "",
        "## Experts:",
        "",
    ]
    
    for expert_name, expert_config in experts.items():
        expert_model = model_registry.get(expert_name)
        description = "No description available"
        if expert_model:
            description = expert_model.description or description
        
        history_mode = expert_config.history
        history_desc = {
            "full": "Full chat history will be provided",
            "condensed": "A condensed summary of chat history will be provided",
            "off": "No chat history will be provided",
        }.get(history_mode, "Unknown history mode")
        
        lines.append(f"### {expert_name}")
        lines.append(f"- **Description**: {description}")
        lines.append(f"- **History Mode**: {history_desc}")
        lines.append("")
    
    lines.extend([
        "## Usage Guidelines:",
        "",
        "1. Analyze the user's query to determine if it matches an expert's specialty.",
        "2. If an expert is better suited, use `call_expert` with a clear, contextual prompt.",
        "3. Provide relevant chat history based on the expert's history mode configuration.",
        "4. Return the expert's response to the user, adding any necessary clarification.",
        "",
    ])
    
    return "\n".join(lines)


def prepare_history_for_expert(
    messages: list[Message],
    history_mode: str,
) -> list[dict[str, str]]:
    """Prepare chat history for an expert based on the history mode.
    
    Args:
        messages: The conversation messages
        history_mode: One of 'full', 'condensed', or 'off'
        
    Returns:
        A list of simplified message dicts for the expert
    """
    if history_mode == "off":
        return []
    
    # Filter to only user/assistant messages (exclude system and tool messages)
    filtered_messages = [
        msg for msg in messages
        if msg.role in ("user", "assistant") and msg.content
    ]
    
    if history_mode == "full":
        return [
            {"role": msg.role, "content": msg.content}
            for msg in filtered_messages
        ]
    
    if history_mode == "condensed":
        # For condensed mode, we keep the last few messages and summarize older ones
        # Keep the last 3 exchanges (6 messages) in full
        max_recent = 6
        
        if len(filtered_messages) <= max_recent:
            return [
                {"role": msg.role, "content": msg.content}
                for msg in filtered_messages
            ]
        
        # Summarize older messages
        older_messages = filtered_messages[:-max_recent]
        recent_messages = filtered_messages[-max_recent:]
        
        # Create a simple summary of older conversation
        summary_parts = []
        for msg in older_messages:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            summary_parts.append(f"{role}: {content}")
        
        summary = "[Earlier conversation summary]\n" + "\n".join(summary_parts)
        
        result = [{"role": "assistant", "content": summary}]
        result.extend([
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ])
        
        return result
    
    # Unknown mode, return empty
    logger.warning(f"Unknown history mode: {history_mode}")
    return []


async def execute_expert_call(
    expert_name: str,
    prompt: str,
    chat_history: list[dict[str, str]] | None,
    model_registry: "Any",
    process_chat_completion_func: "Any",
) -> str:
    """Execute a call to an expert model.
    
    Args:
        expert_name: The name of the expert model to call
        prompt: The prompt to send to the expert
        chat_history: Optional structured chat history
        model_registry: The model registry to look up the expert
        process_chat_completion_func: Function to process chat completions
        
    Returns:
        The expert's response as a string
    """
    from .schemas import ChatCompletionRequest, Message
    
    # Get the expert model config
    expert_config = model_registry.get(expert_name)
    if not expert_config:
        return json.dumps({
            "error": f"Expert model '{expert_name}' not found",
            "available_experts": [m.name for m in model_registry.list_models()],
        })
    
    # Build messages for the expert
    messages: list[Message] = []
    
    # Add chat history if provided
    if chat_history:
        for hist_msg in chat_history:
            messages.append(Message(
                role=hist_msg.get("role", "user"),
                content=hist_msg.get("content", ""),
            ))
    
    # Add the main prompt as a user message
    messages.append(Message(role="user", content=prompt))
    
    # Create the request
    request = ChatCompletionRequest(
        model=expert_name,
        messages=messages,
    )
    
    try:
        # Process the request through the normal flow
        response = await process_chat_completion_func(request, expert_config)
        
        # Extract the response content
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            return json.dumps({"error": "Expert returned empty response"})
            
    except Exception as e:
        logger.exception(f"Error calling expert '{expert_name}'")
        return json.dumps({"error": f"Expert call failed: {str(e)}"})
