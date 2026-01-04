"""Tool handlers for proxy-managed tools."""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

# Registry of tool handlers
_handlers: dict[str, "ToolHandler"] = {}


class ToolHandler(ABC):
    """Base class for tool handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool this handler handles."""
        ...

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool with the given arguments."""
        ...


class GetCurrentTimeHandler(ToolHandler):
    """Handler for getting the current time."""

    @property
    def name(self) -> str:
        return "get_current_time"

    async def execute(self, arguments: dict[str, Any]) -> str:
        timezone_str = arguments.get("timezone", "UTC")
        now = datetime.now(timezone.utc)
        return json.dumps({
            "time": now.isoformat(),
            "timezone": timezone_str,
            "unix_timestamp": int(now.timestamp()),
            "day": now.strftime("%A"),
        })


class CalculatorHandler(ToolHandler):
    """Handler for basic calculations."""

    @property
    def name(self) -> str:
        return "calculator"

    async def execute(self, arguments: dict[str, Any]) -> str:
        expression = arguments.get("expression", "")
        try:
            # Safe evaluation of mathematical expressions
            # Only allow basic math operations
            allowed_chars = set("0123456789+-*/().% ")
            if not all(c in allowed_chars for c in expression):
                return json.dumps({"error": "Invalid characters in expression"})

            result = eval(expression, {"__builtins__": {}}, {})
            return json.dumps({"expression": expression, "result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})


def register_handler(handler: ToolHandler) -> None:
    """Register a tool handler."""
    _handlers[handler.name] = handler


def get_handler(name: str) -> ToolHandler | None:
    """Get a tool handler by name."""
    return _handlers.get(name)


def get_all_handlers() -> dict[str, ToolHandler]:
    """Get all registered handlers."""
    return _handlers.copy()


# Register default handlers
register_handler(GetCurrentTimeHandler())
register_handler(CalculatorHandler())
