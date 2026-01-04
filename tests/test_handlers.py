"""Tests for handlers.py - Tool handlers for proxy-managed tools."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from openai_proxy.handlers import (
    CalculatorHandler,
    GetCurrentTimeHandler,
    ToolHandler,
    get_all_handlers,
    get_handler,
    register_handler,
)


class TestGetCurrentTimeHandler:
    """Tests for GetCurrentTimeHandler."""

    @pytest.fixture
    def handler(self) -> GetCurrentTimeHandler:
        """Create a GetCurrentTimeHandler instance."""
        return GetCurrentTimeHandler()

    def test_handler_name(self, handler):
        """Test that handler name is correct."""
        assert handler.name == "get_current_time"

    @pytest.mark.asyncio
    async def test_execute_returns_json(self, handler):
        """Test that execute returns valid JSON."""
        result = await handler.execute({})
        data = json.loads(result)
        
        assert "time" in data
        assert "timezone" in data
        assert "unix_timestamp" in data
        assert "day" in data

    @pytest.mark.asyncio
    async def test_execute_default_timezone(self, handler):
        """Test execute with default timezone."""
        result = await handler.execute({})
        data = json.loads(result)
        
        assert data["timezone"] == "UTC"

    @pytest.mark.asyncio
    async def test_execute_custom_timezone(self, handler):
        """Test execute with custom timezone argument."""
        result = await handler.execute({"timezone": "America/New_York"})
        data = json.loads(result)
        
        assert data["timezone"] == "America/New_York"

    @pytest.mark.asyncio
    async def test_execute_returns_valid_timestamp(self, handler):
        """Test that unix_timestamp is a valid timestamp."""
        result = await handler.execute({})
        data = json.loads(result)
        
        # Verify it's a reasonable timestamp (after year 2020)
        assert data["unix_timestamp"] > 1577836800

    @pytest.mark.asyncio
    async def test_execute_returns_day_of_week(self, handler):
        """Test that day is a valid day name."""
        result = await handler.execute({})
        data = json.loads(result)
        
        valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert data["day"] in valid_days


class TestCalculatorHandler:
    """Tests for CalculatorHandler."""

    @pytest.fixture
    def handler(self) -> CalculatorHandler:
        """Create a CalculatorHandler instance."""
        return CalculatorHandler()

    def test_handler_name(self, handler):
        """Test that handler name is correct."""
        assert handler.name == "calculator"

    @pytest.mark.asyncio
    async def test_simple_addition(self, handler):
        """Test simple addition."""
        result = await handler.execute({"expression": "2+2"})
        data = json.loads(result)
        
        assert data["expression"] == "2+2"
        assert data["result"] == 4

    @pytest.mark.asyncio
    async def test_subtraction(self, handler):
        """Test subtraction."""
        result = await handler.execute({"expression": "10-3"})
        data = json.loads(result)
        
        assert data["result"] == 7

    @pytest.mark.asyncio
    async def test_multiplication(self, handler):
        """Test multiplication."""
        result = await handler.execute({"expression": "6*7"})
        data = json.loads(result)
        
        assert data["result"] == 42

    @pytest.mark.asyncio
    async def test_division(self, handler):
        """Test division."""
        result = await handler.execute({"expression": "15/3"})
        data = json.loads(result)
        
        assert data["result"] == 5.0

    @pytest.mark.asyncio
    async def test_complex_expression(self, handler):
        """Test complex expression with parentheses."""
        result = await handler.execute({"expression": "(2+3)*4"})
        data = json.loads(result)
        
        assert data["result"] == 20

    @pytest.mark.asyncio
    async def test_modulo(self, handler):
        """Test modulo operation."""
        result = await handler.execute({"expression": "17%5"})
        data = json.loads(result)
        
        assert data["result"] == 2

    @pytest.mark.asyncio
    async def test_decimal_numbers(self, handler):
        """Test decimal numbers."""
        result = await handler.execute({"expression": "3.14*2"})
        data = json.loads(result)
        
        assert data["result"] == pytest.approx(6.28)

    @pytest.mark.asyncio
    async def test_spaces_in_expression(self, handler):
        """Test expression with spaces."""
        result = await handler.execute({"expression": "10 + 5 * 2"})
        data = json.loads(result)
        
        assert data["result"] == 20

    @pytest.mark.asyncio
    async def test_invalid_characters_rejected(self, handler):
        """Test that invalid characters are rejected."""
        result = await handler.execute({"expression": "import os"})
        data = json.loads(result)
        
        assert "error" in data
        assert "Invalid characters" in data["error"]

    @pytest.mark.asyncio
    async def test_letters_rejected(self, handler):
        """Test that letters are rejected."""
        result = await handler.execute({"expression": "abc + 1"})
        data = json.loads(result)
        
        assert "error" in data

    @pytest.mark.asyncio
    async def test_special_chars_rejected(self, handler):
        """Test that special characters are rejected."""
        result = await handler.execute({"expression": "__import__('os')"})
        data = json.loads(result)
        
        assert "error" in data

    @pytest.mark.asyncio
    async def test_division_by_zero_error(self, handler):
        """Test division by zero returns error."""
        result = await handler.execute({"expression": "1/0"})
        data = json.loads(result)
        
        assert "error" in data

    @pytest.mark.asyncio
    async def test_empty_expression(self, handler):
        """Test empty expression."""
        result = await handler.execute({"expression": ""})
        data = json.loads(result)
        
        # Empty string evaluates to error or empty result
        # The implementation should handle this gracefully
        assert "error" in data or "result" in data


class TestHandlerRegistry:
    """Tests for handler registration and retrieval."""

    def test_get_handler_current_time(self):
        """Test getting the current time handler."""
        handler = get_handler("get_current_time")
        assert handler is not None
        assert handler.name == "get_current_time"

    def test_get_handler_calculator(self):
        """Test getting the calculator handler."""
        handler = get_handler("calculator")
        assert handler is not None
        assert handler.name == "calculator"

    def test_get_handler_nonexistent(self):
        """Test getting a non-existent handler returns None."""
        handler = get_handler("nonexistent_tool")
        assert handler is None

    def test_get_all_handlers(self):
        """Test getting all registered handlers."""
        handlers = get_all_handlers()
        
        assert "get_current_time" in handlers
        assert "calculator" in handlers
        assert isinstance(handlers["get_current_time"], GetCurrentTimeHandler)
        assert isinstance(handlers["calculator"], CalculatorHandler)

    def test_get_all_handlers_returns_copy(self):
        """Test that get_all_handlers returns a copy."""
        handlers1 = get_all_handlers()
        handlers2 = get_all_handlers()
        
        # Modifying one shouldn't affect the other
        handlers1["test"] = "value"
        assert "test" not in handlers2


class TestCustomHandler:
    """Tests for registering custom handlers."""

    class MockHandler(ToolHandler):
        """A mock tool handler for testing."""

        @property
        def name(self) -> str:
            return "mock_tool"

        async def execute(self, arguments: dict) -> str:
            return json.dumps({"mock": True, "args": arguments})

    def test_register_custom_handler(self):
        """Test registering a custom handler."""
        handler = self.MockHandler()
        register_handler(handler)
        
        retrieved = get_handler("mock_tool")
        assert retrieved is not None
        assert retrieved.name == "mock_tool"

    @pytest.mark.asyncio
    async def test_custom_handler_execute(self):
        """Test executing a custom handler."""
        handler = self.MockHandler()
        register_handler(handler)
        
        retrieved = get_handler("mock_tool")
        result = await retrieved.execute({"test": "value"})
        data = json.loads(result)
        
        assert data["mock"] is True
        assert data["args"]["test"] == "value"

    def test_register_overwrites_existing(self):
        """Test that registering with same name overwrites."""
        handler1 = self.MockHandler()
        register_handler(handler1)
        
        handler2 = self.MockHandler()
        register_handler(handler2)
        
        # Should return the second handler
        retrieved = get_handler("mock_tool")
        assert retrieved is handler2
