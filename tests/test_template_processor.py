"""Tests for template_processor.py - Template processing for system prompts."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from openai_proxy.template_processor import (
    TEMPLATE_PATTERN,
    call_tool,
    find_templates,
    parse_arguments,
    process_system_prompt,
    process_template,
)


class TestParseArguments:
    """Tests for parse_arguments function."""

    def test_empty_string(self):
        """Test parsing empty string."""
        result = parse_arguments("")
        assert result == {}

    def test_whitespace_only(self):
        """Test parsing whitespace only."""
        result = parse_arguments("   ")
        assert result == {}

    def test_single_string_double_quotes(self):
        """Test parsing single string argument with double quotes."""
        result = parse_arguments('timezone="UTC"')
        assert result == {"timezone": "UTC"}

    def test_single_string_single_quotes(self):
        """Test parsing single string argument with single quotes."""
        result = parse_arguments("timezone='UTC'")
        assert result == {"timezone": "UTC"}

    def test_integer_value(self):
        """Test parsing integer value."""
        result = parse_arguments("count=42")
        assert result == {"count": 42}
        assert isinstance(result["count"], int)

    def test_float_value(self):
        """Test parsing float value."""
        result = parse_arguments("temperature=0.7")
        assert result == {"temperature": 0.7}
        assert isinstance(result["temperature"], float)

    def test_boolean_true(self):
        """Test parsing boolean true."""
        result = parse_arguments("enabled=true")
        assert result == {"enabled": True}
        assert isinstance(result["enabled"], bool)

    def test_boolean_false(self):
        """Test parsing boolean false."""
        result = parse_arguments("enabled=false")
        assert result == {"enabled": False}

    def test_null_value(self):
        """Test parsing null value."""
        result = parse_arguments("value=null")
        assert result == {"value": None}

    def test_multiple_arguments(self):
        """Test parsing multiple arguments."""
        result = parse_arguments('name="test", count=5, enabled=true')
        
        assert result == {
            "name": "test",
            "count": 5,
            "enabled": True,
        }

    def test_arguments_with_spaces(self):
        """Test parsing arguments with spaces."""
        result = parse_arguments('  name = "test" ,  count = 10  ')
        
        assert result["name"] == "test"
        assert result["count"] == 10

    def test_string_with_spaces(self):
        """Test parsing string value with spaces."""
        result = parse_arguments('message="Hello World"')
        assert result == {"message": "Hello World"}

    def test_empty_string_value(self):
        """Test parsing empty string value."""
        result = parse_arguments('value=""')
        assert result == {"value": ""}


class TestTemplatePattern:
    """Tests for the TEMPLATE_PATTERN regex."""

    def test_simple_template_no_args(self):
        """Test matching simple template with no arguments."""
        match = TEMPLATE_PATTERN.search("{{get_time()}}")
        
        assert match is not None
        assert match.group(1) == "get_time"
        assert match.group(2) == ""

    def test_template_with_args(self):
        """Test matching template with arguments."""
        match = TEMPLATE_PATTERN.search('{{get_time(timezone="UTC")}}')
        
        assert match is not None
        assert match.group(1) == "get_time"
        assert match.group(2) == 'timezone="UTC"'

    def test_template_with_spaces(self):
        """Test matching template with whitespace."""
        match = TEMPLATE_PATTERN.search('{{ get_time( timezone="UTC" ) }}')
        
        assert match is not None
        assert match.group(1) == "get_time"

    def test_template_in_text(self):
        """Test finding template within text."""
        text = "The current time is: {{get_time()}}. Have a nice day!"
        match = TEMPLATE_PATTERN.search(text)
        
        assert match is not None
        assert match.group(1) == "get_time"

    def test_multiple_templates(self):
        """Test finding multiple templates."""
        text = "Time: {{get_time()}} Calc: {{calculator(expression='2+2')}}"
        matches = list(TEMPLATE_PATTERN.finditer(text))
        
        assert len(matches) == 2
        assert matches[0].group(1) == "get_time"
        assert matches[1].group(1) == "calculator"

    def test_no_template(self):
        """Test text without templates."""
        text = "This is plain text without templates."
        match = TEMPLATE_PATTERN.search(text)
        
        assert match is None

    def test_mcp_tool_template(self):
        """Test matching MCP tool template."""
        match = TEMPLATE_PATTERN.search('{{mcp_server_tool_name()}}')
        
        assert match is not None
        assert match.group(1) == "mcp_server_tool_name"


class TestFindTemplates:
    """Tests for find_templates function."""

    def test_find_no_templates(self):
        """Test finding no templates in plain text."""
        templates = find_templates("Plain text without templates")
        assert templates == []

    def test_find_templates_empty_string(self):
        """Test with empty string."""
        templates = find_templates("")
        assert templates == []

    def test_find_templates_none(self):
        """Test with None."""
        templates = find_templates(None)
        assert templates == []

    def test_find_single_template(self):
        """Test finding a single template."""
        templates = find_templates("Time: {{get_time()}}")
        
        assert len(templates) == 1
        assert templates[0]["tool_name"] == "get_time"
        assert templates[0]["arguments"] == {}

    def test_find_template_with_args(self):
        """Test finding template with arguments."""
        templates = find_templates('{{get_time(timezone="UTC")}}')
        
        assert len(templates) == 1
        assert templates[0]["tool_name"] == "get_time"
        assert templates[0]["arguments"] == {"timezone": "UTC"}

    def test_find_multiple_templates(self):
        """Test finding multiple templates."""
        prompt = """
        Current time: {{get_time()}}
        Calculation: {{calculator(expression="2+2")}}
        """
        templates = find_templates(prompt)
        
        assert len(templates) == 2
        
        names = [t["tool_name"] for t in templates]
        assert "get_time" in names
        assert "calculator" in names

    def test_find_templates_returns_positions(self):
        """Test that find_templates returns positions."""
        templates = find_templates("Start {{get_time()}} End")
        
        assert len(templates) == 1
        assert "start" in templates[0]
        assert "end" in templates[0]
        assert templates[0]["start"] > 0
        assert templates[0]["end"] > templates[0]["start"]


class TestCallTool:
    """Tests for call_tool function."""

    @pytest.mark.asyncio
    async def test_call_builtin_handler(self):
        """Test calling a built-in handler."""
        result = await call_tool("get_current_time", {})
        
        # Should return valid JSON
        data = json.loads(result)
        assert "time" in data
        assert "timezone" in data

    @pytest.mark.asyncio
    async def test_call_calculator(self):
        """Test calling calculator handler."""
        result = await call_tool("calculator", {"expression": "2+2"})
        
        data = json.loads(result)
        assert data["result"] == 4

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        """Test calling unknown tool."""
        result = await call_tool("unknown_tool", {})
        
        assert "[Unknown tool: unknown_tool]" in result

    @pytest.mark.asyncio
    async def test_call_mcp_tool(self):
        """Test calling an MCP tool."""
        mock_mcp_client = AsyncMock()
        mock_mcp_client.call_tool.return_value = {"status": "success"}
        
        with patch("openai_proxy.template_processor.get_mcp_client", return_value=mock_mcp_client):
            result = await call_tool("mcp_server_tool", {"arg": "value"})
        
        data = json.loads(result)
        assert data["status"] == "success"
        mock_mcp_client.call_tool.assert_called_once()


class TestProcessTemplate:
    """Tests for process_template function."""

    @pytest.mark.asyncio
    async def test_process_simple_template(self):
        """Test processing a simple template."""
        result = await process_template("get_current_time()")
        
        # Should return valid JSON from the handler
        data = json.loads(result)
        assert "time" in data

    @pytest.mark.asyncio
    async def test_process_template_with_args(self):
        """Test processing template with arguments."""
        result = await process_template('calculator(expression="3*4")')
        
        data = json.loads(result)
        assert data["result"] == 12

    @pytest.mark.asyncio
    async def test_process_invalid_template(self):
        """Test processing invalid template."""
        # Invalid template format
        result = await process_template("not a valid template")
        
        assert "[Invalid template:" in result


class TestProcessSystemPrompt:
    """Tests for process_system_prompt function."""

    @pytest.mark.asyncio
    async def test_process_prompt_no_templates(self):
        """Test processing prompt without templates."""
        prompt = "You are a helpful assistant."
        result = await process_system_prompt(prompt)
        
        assert result == prompt

    @pytest.mark.asyncio
    async def test_process_prompt_with_template(self):
        """Test processing prompt with a template."""
        prompt = "Current time: {{get_current_time()}}"
        result = await process_system_prompt(prompt)
        
        # Template should be replaced with actual time data
        assert "{{" not in result
        assert "time" in result  # Should contain time info

    @pytest.mark.asyncio
    async def test_process_prompt_preserves_text(self):
        """Test that non-template text is preserved."""
        prompt = "Hello! {{calculator(expression='1+1')}} Goodbye!"
        result = await process_system_prompt(prompt)
        
        assert "Hello!" in result
        assert "Goodbye!" in result
        assert "{{" not in result

    @pytest.mark.asyncio
    async def test_process_prompt_multiple_templates(self):
        """Test processing prompt with multiple templates."""
        prompt = """
        Time: {{get_current_time()}}
        Math: {{calculator(expression="5+5")}}
        """
        result = await process_system_prompt(prompt)
        
        assert "{{" not in result
        assert "Time:" in result
        assert "Math:" in result

    @pytest.mark.asyncio
    async def test_process_prompt_none(self):
        """Test processing None prompt."""
        result = await process_system_prompt(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_process_prompt_empty(self):
        """Test processing empty prompt."""
        result = await process_system_prompt("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_process_prompt_concurrent_templates(self):
        """Test that multiple templates are processed concurrently."""
        prompt = "{{calculator(expression='1+1')}} {{calculator(expression='2+2')}}"
        result = await process_system_prompt(prompt)
        
        # Both should be processed
        assert "{{" not in result
        # Results should be present (order might vary)
        assert "2" in result  # Result of 1+1
        assert "4" in result  # Result of 2+2


class TestTemplateEdgeCases:
    """Tests for edge cases in template processing."""

    def test_nested_braces_not_matched(self):
        """Test that nested braces don't cause issues."""
        # This should not match as a template
        text = "{{not{a}template}}"
        templates = find_templates(text)
        
        # Should not find invalid templates
        for t in templates:
            assert t["tool_name"].isidentifier()

    def test_incomplete_template(self):
        """Test that incomplete templates are not matched."""
        templates = find_templates("{{incomplete")
        assert templates == []
        
        templates = find_templates("incomplete}}")
        assert templates == []

    @pytest.mark.asyncio
    async def test_template_error_handling(self):
        """Test that template errors don't crash processing."""
        # Mock a handler that raises an exception
        with patch("openai_proxy.template_processor.get_handler") as mock_get:
            mock_handler = AsyncMock()
            mock_handler.name = "failing_tool"
            mock_handler.execute.side_effect = Exception("Test error")
            mock_get.return_value = mock_handler
            
            result = await call_tool("failing_tool", {})
            
            assert "[Error" in result

    def test_template_with_special_characters_in_string(self):
        """Test template with special characters in string argument."""
        templates = find_templates('{{tool(path="/path/to/file")}}')
        
        assert len(templates) == 1
        assert templates[0]["arguments"]["path"] == "/path/to/file"

    @pytest.mark.asyncio
    async def test_process_preserves_template_order(self):
        """Test that template results are in correct positions."""
        prompt = "A {{calculator(expression='1+1')}} B {{calculator(expression='2+2')}} C"
        result = await process_system_prompt(prompt)
        
        # Check order is preserved
        a_pos = result.index("A")
        b_pos = result.index("B")
        c_pos = result.index("C")
        
        assert a_pos < b_pos < c_pos
