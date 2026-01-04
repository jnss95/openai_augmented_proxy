"""Tests for template_processor.py - Jinja2-based template processing."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from openai_proxy.template_processor import (
    TemplateProcessor,
    ToolCallExtension,
    call_tool,
    get_template_processor,
    initialize_template_processor,
    process_system_prompt,
)
from openai_proxy.variable_store import VariableStore, YamlVariableBackend, set_variable_store


class TestToolCallExtension:
    """Tests for ToolCallExtension class."""

    def test_create_placeholder(self):
        """Test creating placeholders for tool calls."""
        ext = ToolCallExtension()
        
        placeholder1 = ext.create_placeholder("tool1", {"arg": "value"})
        placeholder2 = ext.create_placeholder("tool2", {})
        
        assert placeholder1 != placeholder2
        assert "__TOOL_PLACEHOLDER_" in placeholder1
        assert "__TOOL_PLACEHOLDER_" in placeholder2

    def test_pending_calls_tracked(self):
        """Test that pending calls are tracked."""
        ext = ToolCallExtension()
        
        ext.create_placeholder("tool1", {"arg": "value"})
        ext.create_placeholder("tool2", {"x": 1})
        
        assert len(ext.pending_calls) == 2
        assert ext.pending_calls[0][0] == "tool1"
        assert ext.pending_calls[1][0] == "tool2"


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


class TestTemplateProcessor:
    """Tests for TemplateProcessor class."""

    @pytest.fixture
    def empty_store(self):
        """Create an empty variable store."""
        store = VariableStore()
        set_variable_store(store)
        return store

    @pytest.fixture
    def store_with_vars(self, tmp_path):
        """Create a variable store with some variables."""
        vars_file = tmp_path / "variables.yaml"
        vars_file.write_text("""
app_name: "Test App"
version: "1.0"
settings:
  max_tokens: 100
  debug: true
items:
  - apple
  - banana
  - cherry
""")
        store = VariableStore()
        store.add_backend(YamlVariableBackend(vars_file))
        set_variable_store(store)
        return store

    @pytest.mark.asyncio
    async def test_process_plain_text(self, empty_store):
        """Test processing plain text without templates."""
        processor = TemplateProcessor()
        result = await processor.process("Hello, world!")
        
        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_process_with_variable(self, store_with_vars):
        """Test processing template with variables."""
        processor = TemplateProcessor()
        result = await processor.process("App: {{ app_name }}")
        
        assert result == "App: Test App"

    @pytest.mark.asyncio
    async def test_process_nested_variable(self, store_with_vars):
        """Test processing template with nested variables."""
        processor = TemplateProcessor()
        result = await processor.process("Tokens: {{ settings.max_tokens }}")
        
        assert result == "Tokens: 100"

    @pytest.mark.asyncio
    async def test_process_with_conditional(self, store_with_vars):
        """Test processing template with conditionals."""
        processor = TemplateProcessor()
        template = "{% if settings.debug %}Debug ON{% else %}Debug OFF{% endif %}"
        result = await processor.process(template)
        
        assert result == "Debug ON"

    @pytest.mark.asyncio
    async def test_process_with_loop(self, store_with_vars):
        """Test processing template with loops."""
        processor = TemplateProcessor()
        template = "{% for item in items %}{{ item }} {% endfor %}"
        result = await processor.process(template)
        
        assert "apple" in result
        assert "banana" in result
        assert "cherry" in result

    @pytest.mark.asyncio
    async def test_process_with_filter(self, store_with_vars):
        """Test processing template with filters."""
        processor = TemplateProcessor()
        result = await processor.process("{{ app_name | upper }}")
        
        assert result == "TEST APP"

    @pytest.mark.asyncio
    async def test_process_with_default_filter(self, empty_store):
        """Test processing template with default filter for missing var."""
        processor = TemplateProcessor()
        result = await processor.process("{{ missing | default('fallback') }}")
        
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_process_with_tool_call(self, empty_store):
        """Test processing template with tool call."""
        processor = TemplateProcessor()
        result = await processor.process('Time: {{ tool("get_current_time") }}')
        
        # Should contain the tool result (JSON with time)
        assert "time" in result
        assert "timezone" in result

    @pytest.mark.asyncio
    async def test_process_with_tool_call_args(self, empty_store):
        """Test processing template with tool call with arguments."""
        processor = TemplateProcessor()
        result = await processor.process('{{ tool("calculator", expression="3*4") }}')
        
        data = json.loads(result)
        assert data["result"] == 12

    @pytest.mark.asyncio
    async def test_process_multiple_tool_calls(self, empty_store):
        """Test processing template with multiple tool calls."""
        processor = TemplateProcessor()
        template = '{{ tool("calculator", expression="1+1") }} and {{ tool("calculator", expression="2+2") }}'
        result = await processor.process(template)
        
        # Both results should be present
        assert "2" in result  # 1+1
        assert "4" in result  # 2+2

    @pytest.mark.asyncio
    async def test_process_with_extra_variables(self, empty_store):
        """Test processing template with extra variables."""
        processor = TemplateProcessor()
        result = await processor.process(
            "Hello, {{ name }}!",
            extra_variables={"name": "World"}
        )
        
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_process_extra_vars_override_store(self, store_with_vars):
        """Test that extra variables override store variables."""
        processor = TemplateProcessor()
        result = await processor.process(
            "{{ app_name }}",
            extra_variables={"app_name": "Override App"}
        )
        
        assert result == "Override App"


class TestTemplateProcessorWithFiles:
    """Tests for TemplateProcessor with template files."""

    @pytest.fixture
    def templates_dir(self, tmp_path):
        """Create a templates directory with test templates."""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        
        # Create a simple template
        (tpl_dir / "simple.j2").write_text("Simple template content")
        
        # Create a template with variables
        (tpl_dir / "with-vars.j2").write_text("App: {{ app_name }}")
        
        # Create a template that includes another
        (tpl_dir / "parent.j2").write_text('{% include "simple.j2" %} - Extended')
        
        # Create subdirectory
        (tpl_dir / "common").mkdir()
        (tpl_dir / "common" / "header.j2").write_text("=== HEADER ===")
        
        return tpl_dir

    @pytest.fixture
    def store_with_vars(self, tmp_path):
        """Create a variable store with some variables."""
        vars_file = tmp_path / "variables.yaml"
        vars_file.write_text("app_name: Test App")
        store = VariableStore()
        store.add_backend(YamlVariableBackend(vars_file))
        set_variable_store(store)
        return store

    @pytest.mark.asyncio
    async def test_process_file(self, templates_dir, store_with_vars):
        """Test processing a template file."""
        processor = TemplateProcessor(templates_dir)
        result = await processor.process_file("simple.j2")
        
        assert result == "Simple template content"

    @pytest.mark.asyncio
    async def test_process_file_with_vars(self, templates_dir, store_with_vars):
        """Test processing a template file with variables."""
        processor = TemplateProcessor(templates_dir)
        result = await processor.process_file("with-vars.j2")
        
        assert result == "App: Test App"

    @pytest.mark.asyncio
    async def test_include_template(self, templates_dir, store_with_vars):
        """Test including templates."""
        processor = TemplateProcessor(templates_dir)
        result = await processor.process(
            '{% include "simple.j2" %} - inline'
        )
        
        assert result == "Simple template content - inline"

    @pytest.mark.asyncio
    async def test_include_nested_template(self, templates_dir, store_with_vars):
        """Test including templates from subdirectories."""
        processor = TemplateProcessor(templates_dir)
        result = await processor.process(
            '{% include "common/header.j2" %}'
        )
        
        assert result == "=== HEADER ==="

    @pytest.mark.asyncio
    async def test_file_not_found(self, templates_dir, store_with_vars):
        """Test error handling for missing template file."""
        processor = TemplateProcessor(templates_dir)
        
        with pytest.raises(FileNotFoundError):
            await processor.process_file("nonexistent.j2")


class TestProcessSystemPrompt:
    """Tests for process_system_prompt function."""

    @pytest.fixture(autouse=True)
    def setup_empty_store(self):
        """Set up an empty variable store for each test."""
        store = VariableStore()
        set_variable_store(store)

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
    async def test_process_prompt_plain_text(self):
        """Test processing plain text prompt."""
        result = await process_system_prompt("You are a helpful assistant.")
        assert result == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_process_prompt_with_tool(self):
        """Test processing prompt with tool call."""
        # Initialize the processor
        initialize_template_processor(None)
        
        prompt = 'Current time: {{ tool("get_current_time") }}'
        result = await process_system_prompt(prompt)
        
        assert "time" in result


class TestGlobalFunctions:
    """Tests for global template processor functions."""

    def test_get_template_processor_creates_instance(self):
        """Test that get_template_processor creates an instance."""
        processor = get_template_processor()
        assert processor is not None
        assert isinstance(processor, TemplateProcessor)

    def test_initialize_template_processor(self, tmp_path):
        """Test initializing template processor with directory."""
        processor = initialize_template_processor(tmp_path)
        
        assert processor.templates_dir == tmp_path

    def test_initialize_template_processor_none(self):
        """Test initializing template processor without directory."""
        processor = initialize_template_processor(None)
        
        assert processor.templates_dir is None
