"""Tests for models.py - Model configuration schema and loader."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from openai_proxy.models import (
    ExpertConfig,
    ModelConfig,
    ModelRegistry,
    Tool,
    ToolFunction,
    ToolParameter,
    get_model_registry,
    reload_model_registry,
)


class TestToolParameter:
    """Tests for ToolParameter schema."""

    def test_default_values(self):
        """Test default parameter values."""
        param = ToolParameter()
        
        assert param.type == "string"
        assert param.description == ""
        assert param.enum is None
        assert param.default is None

    def test_custom_values(self):
        """Test creating parameter with custom values."""
        param = ToolParameter(
            type="integer",
            description="A number",
            enum=None,
            default=10,
        )
        
        assert param.type == "integer"
        assert param.description == "A number"
        assert param.default == 10

    def test_enum_values(self):
        """Test parameter with enum values."""
        param = ToolParameter(
            type="string",
            description="Choose a color",
            enum=["red", "green", "blue"],
        )
        
        assert param.enum == ["red", "green", "blue"]


class TestToolFunction:
    """Tests for ToolFunction schema."""

    def test_minimal_function(self):
        """Test creating minimal function."""
        func = ToolFunction(name="test_func")
        
        assert func.name == "test_func"
        assert func.description == ""
        assert func.parameters == {"type": "object", "properties": {}, "required": []}

    def test_function_with_parameters(self):
        """Test function with parameters."""
        func = ToolFunction(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        )
        
        assert func.name == "get_weather"
        assert func.description == "Get weather for a location"
        assert "location" in func.parameters["properties"]


class TestTool:
    """Tests for Tool schema."""

    def test_tool_creation(self):
        """Test creating a tool."""
        tool = Tool(
            type="function",
            function=ToolFunction(name="test_tool", description="A test")
        )
        
        assert tool.type == "function"
        assert tool.function.name == "test_tool"
        assert tool.handler is None

    def test_tool_default_type(self):
        """Test that type defaults to function."""
        tool = Tool(function=ToolFunction(name="test"))
        assert tool.type == "function"

    def test_tool_with_handler(self):
        """Test tool with handler reference."""
        tool = Tool(
            function=ToolFunction(name="test"),
            handler="custom_handler"
        )
        
        assert tool.handler == "custom_handler"

    def test_handler_excluded_from_export(self):
        """Test that handler is excluded when exporting."""
        tool = Tool(
            function=ToolFunction(name="test"),
            handler="my_handler"
        )
        
        data = tool.model_dump()
        assert "handler" not in data


class TestExpertConfig:
    """Tests for ExpertConfig schema."""

    def test_default_values(self):
        """Test default expert config values."""
        config = ExpertConfig()
        
        assert config.history == "full"

    def test_custom_history_full(self):
        """Test expert with full history mode."""
        config = ExpertConfig(history="full")
        assert config.history == "full"

    def test_custom_history_condensed(self):
        """Test expert with condensed history mode."""
        config = ExpertConfig(history="condensed")
        assert config.history == "condensed"

    def test_custom_history_off(self):
        """Test expert with history off."""
        config = ExpertConfig(history="off")
        assert config.history == "off"

    def test_history_false_converts_to_off(self):
        """Test that False is converted to 'off' (YAML parses 'off' as False)."""
        config = ExpertConfig(history=False)
        assert config.history == "off"

    def test_history_true_converts_to_full(self):
        """Test that True is converted to 'full'."""
        config = ExpertConfig(history=True)
        assert config.history == "full"


class TestModelConfig:
    """Tests for ModelConfig schema."""

    def test_minimal_config(self):
        """Test minimal model configuration."""
        config = ModelConfig(name="test-model")
        
        assert config.name == "test-model"
        assert config.upstream_model is None
        assert config.system_prompt is None
        assert config.tools == []
        assert config.mcp_servers == []
        assert config.skills == []
        assert config.include_global_skills is True
        assert config.owned_by == "openai-proxy"

    def test_full_config(self):
        """Test full model configuration."""
        config = ModelConfig(
            name="assistant",
            upstream_model="gpt-4o-mini",
            system_prompt="You are a helpful assistant.",
            tools=[
                Tool(function=ToolFunction(name="get_time"))
            ],
            mcp_servers=["homeassistant"],
            skills=["code-review"],
            include_global_skills=False,
            description="An assistant model",
            created=1700000000,
            owned_by="custom",
        )
        
        assert config.name == "assistant"
        assert config.upstream_model == "gpt-4o-mini"
        assert config.system_prompt == "You are a helpful assistant."
        assert len(config.tools) == 1
        assert config.mcp_servers == ["homeassistant"]
        assert config.skills == ["code-review"]
        assert config.include_global_skills is False
        assert config.owned_by == "custom"

    def test_effective_upstream_model_with_upstream(self):
        """Test effective_upstream_model when upstream_model is set."""
        config = ModelConfig(name="my-model", upstream_model="gpt-4")
        assert config.effective_upstream_model == "gpt-4"

    def test_effective_upstream_model_without_upstream(self):
        """Test effective_upstream_model when upstream_model is not set."""
        config = ModelConfig(name="my-model")
        assert config.effective_upstream_model == "my-model"

    def test_mcp_servers_as_list(self):
        """Test mcp_servers as a simple list."""
        config = ModelConfig(
            name="test",
            mcp_servers=["server1", "server2"]
        )
        
        assert config.mcp_servers == ["server1", "server2"]

    def test_mcp_servers_as_dict(self):
        """Test mcp_servers as dict with filters."""
        config = ModelConfig(
            name="test",
            mcp_servers={
                "server1": {"whitelist": ["tool1", "tool2"]},
                "server2": {"blacklist": ["tool3"]}
            }
        )
        
        assert "server1" in config.mcp_servers
        assert config.mcp_servers["server1"]["whitelist"] == ["tool1", "tool2"]

    def test_experts_empty_by_default(self):
        """Test that experts is empty by default."""
        config = ModelConfig(name="test")
        assert config.experts == {}

    def test_experts_configuration(self):
        """Test experts configuration."""
        config = ModelConfig(
            name="router",
            experts={
                "augmented/perplexity": ExpertConfig(history="condensed"),
                "augmented/homeassistant": ExpertConfig(history="full"),
                "augmented/music": ExpertConfig(history="off"),
            }
        )
        
        assert len(config.experts) == 3
        assert config.experts["augmented/perplexity"].history == "condensed"
        assert config.experts["augmented/homeassistant"].history == "full"
        assert config.experts["augmented/music"].history == "off"


class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture
    def registry(self) -> ModelRegistry:
        """Create a new ModelRegistry instance."""
        return ModelRegistry()

    def test_register_model(self, registry):
        """Test registering a model."""
        config = ModelConfig(name="test-model")
        registry.register(config)
        
        assert registry.get("test-model") is config

    def test_get_nonexistent_model(self, registry):
        """Test getting a non-existent model returns None."""
        assert registry.get("nonexistent") is None

    def test_list_models_empty(self, registry):
        """Test listing models when empty."""
        assert registry.list_models() == []

    def test_list_models(self, registry):
        """Test listing registered models."""
        config1 = ModelConfig(name="model1")
        config2 = ModelConfig(name="model2")
        
        registry.register(config1)
        registry.register(config2)
        
        models = registry.list_models()
        assert len(models) == 2
        assert config1 in models
        assert config2 in models

    def test_register_overwrites(self, registry):
        """Test that registering with same name overwrites."""
        config1 = ModelConfig(name="test", system_prompt="First")
        config2 = ModelConfig(name="test", system_prompt="Second")
        
        registry.register(config1)
        registry.register(config2)
        
        result = registry.get("test")
        assert result.system_prompt == "Second"


class TestModelRegistryFileLoading:
    """Tests for loading model configs from files."""

    @pytest.fixture
    def models_dir(self) -> Path:
        """Create a temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_from_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        registry = ModelRegistry()
        registry.load_from_directory("/nonexistent/path")
        
        # Should not raise, just return empty
        assert registry.list_models() == []

    def test_load_yaml_file(self, models_dir):
        """Test loading a .yaml file."""
        config_data = {
            "name": "yaml-model",
            "upstream_model": "gpt-4",
            "system_prompt": "You are helpful.",
        }
        
        with open(models_dir / "model.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        registry = ModelRegistry()
        registry.load_from_directory(models_dir)
        
        model = registry.get("yaml-model")
        assert model is not None
        assert model.upstream_model == "gpt-4"

    def test_load_yml_file(self, models_dir):
        """Test loading a .yml file."""
        config_data = {
            "name": "yml-model",
            "system_prompt": "Test prompt",
        }
        
        with open(models_dir / "model.yml", "w") as f:
            yaml.dump(config_data, f)
        
        registry = ModelRegistry()
        registry.load_from_directory(models_dir)
        
        model = registry.get("yml-model")
        assert model is not None

    def test_load_multiple_models_from_list(self, models_dir):
        """Test loading multiple models from a single file."""
        config_data = [
            {"name": "model1", "system_prompt": "Prompt 1"},
            {"name": "model2", "system_prompt": "Prompt 2"},
        ]
        
        with open(models_dir / "models.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        registry = ModelRegistry()
        registry.load_from_directory(models_dir)
        
        assert registry.get("model1") is not None
        assert registry.get("model2") is not None

    def test_load_empty_file(self, models_dir):
        """Test loading an empty YAML file."""
        (models_dir / "empty.yaml").touch()
        
        registry = ModelRegistry()
        registry.load_from_directory(models_dir)
        
        # Should not raise, just skip empty file
        assert registry.list_models() == []

    def test_load_invalid_yaml(self, models_dir):
        """Test loading invalid YAML file."""
        with open(models_dir / "invalid.yaml", "w") as f:
            f.write("invalid: yaml: content: [")
        
        registry = ModelRegistry()
        # Should not raise, but print warning
        registry.load_from_directory(models_dir)

    def test_load_model_with_tools(self, models_dir):
        """Test loading model with tool definitions."""
        config_data = {
            "name": "tool-model",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Get current time",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
            ]
        }
        
        with open(models_dir / "tools.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        registry = ModelRegistry()
        registry.load_from_directory(models_dir)
        
        model = registry.get("tool-model")
        assert model is not None
        assert len(model.tools) == 1
        assert model.tools[0].function.name == "get_time"


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_model_registry_creates_singleton(self):
        """Test that get_model_registry creates a singleton."""
        # Reset global state
        import openai_proxy.models as models_module
        models_module._registry = None
        
        with patch("openai_proxy.models.get_settings") as mock_settings:
            mock_settings.return_value.models_config_dir = "/nonexistent"
            
            registry1 = get_model_registry()
            registry2 = get_model_registry()
            
            assert registry1 is registry2

    def test_reload_model_registry(self):
        """Test that reload creates a new registry."""
        import openai_proxy.models as models_module
        models_module._registry = None
        
        with patch("openai_proxy.models.get_settings") as mock_settings:
            mock_settings.return_value.models_config_dir = "/nonexistent"
            
            registry1 = get_model_registry()
            registry2 = reload_model_registry()
            
            assert registry1 is not registry2
