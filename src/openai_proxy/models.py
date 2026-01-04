"""Model configuration schema and loader."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .config import get_settings


class ToolParameter(BaseModel):
    """Schema for a tool parameter."""

    type: str = "string"
    description: str = ""
    enum: list[str] | None = None
    default: Any | None = None


class ToolFunction(BaseModel):
    """Schema for a tool function definition."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})


class Tool(BaseModel):
    """Schema for a tool definition."""

    type: str = "function"
    function: ToolFunction

    # Internal handler reference (not sent to client)
    handler: str | None = Field(default=None, exclude=True)


class ModelConfig(BaseModel):
    """Configuration for a model exposed by the proxy."""

    # The name to expose this model as
    name: str

    # The actual upstream model to use
    upstream_model: str | None = Field(
        default=None,
        description="The actual model name to use when calling the upstream API. If not set, uses 'name'.",
    )

    # System prompt to prepend to all requests
    system_prompt: str | None = Field(
        default=None,
        description="System prompt to prepend to conversations for this model",
    )

    # Additional tools to augment requests with
    tools: list[Tool] = Field(
        default_factory=list,
        description="Additional tools to add to requests for this model",
    )

    # MCP servers to connect and expose tools from
    # Can be a simple list: ["server1", "server2"]
    # Or a dict with per-server filtering:
    #   server1:
    #     whitelist: ["tool1", "tool2"]
    #     blacklist: ["tool3"]
    mcp_servers: list[str] | dict[str, dict[str, list[str]] | None] = Field(
        default_factory=list,
        description="MCP servers to connect. Can be a list or dict with whitelist/blacklist per server",
    )

    # Skills to include in the system prompt
    skills: list[str] = Field(
        default_factory=list,
        description="Names of skills to include in the system prompt",
    )

    # Whether to include global skills (from conf/skills/)
    include_global_skills: bool = Field(
        default=True,
        description="Whether to include global skills in addition to model-specific skills",
    )

    # Model metadata for /v1/models endpoint
    description: str | None = None
    created: int | None = None
    owned_by: str = "openai-proxy"

    @property
    def effective_upstream_model(self) -> str:
        """Get the model name to use for upstream API calls."""
        return self.upstream_model or self.name


class ModelRegistry:
    """Registry for model configurations."""

    def __init__(self):
        self._models: dict[str, ModelConfig] = {}

    def register(self, config: ModelConfig) -> None:
        """Register a model configuration."""
        self._models[config.name] = config

    def get(self, name: str) -> ModelConfig | None:
        """Get a model configuration by name."""
        return self._models.get(name)

    def list_models(self) -> list[ModelConfig]:
        """List all registered models."""
        return list(self._models.values())

    def load_from_directory(self, directory: str | Path) -> None:
        """Load model configurations from YAML files in a directory."""
        directory = Path(directory)
        if not directory.exists():
            return

        for file_path in directory.glob("*.yaml"):
            self._load_file(file_path)
        for file_path in directory.glob("*.yml"):
            self._load_file(file_path)

    def _load_file(self, file_path: Path) -> None:
        """Load a single model configuration file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                return

            # Support single model or list of models in one file
            if isinstance(data, list):
                for item in data:
                    config = ModelConfig.model_validate(item)
                    self.register(config)
            else:
                config = ModelConfig.model_validate(data)
                self.register(config)

        except Exception as e:
            print(f"Warning: Failed to load model config from {file_path}: {e}")


# Global model registry
_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry, loading configs if needed."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        settings = get_settings()
        _registry.load_from_directory(settings.models_config_dir)
    return _registry


def reload_model_registry() -> ModelRegistry:
    """Force reload of the model registry."""
    global _registry
    _registry = None
    return get_model_registry()
