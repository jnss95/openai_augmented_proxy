"""Main configuration for the OpenAI API Proxy."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_default_config_dir() -> str:
    """Return default config directory: $HOME/.config/openai_proxy"""
    return str(Path.home() / ".config" / "openai_proxy")


def _get_env_file() -> str | None:
    """Return .env path only if it exists in current working directory."""
    env_path = Path.cwd() / ".env"
    return str(env_path) if env_path.exists() else None


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    Priority: environment variables > .env file (if exists in cwd)
    """

    model_config = SettingsConfigDict(
        env_file=_get_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Backend OpenAI-compatible API configuration
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL of the upstream OpenAI-compatible API",
    )
    api_key: str = Field(
        default="",
        description="API key for the upstream API",
    )

    # Proxy server configuration
    host: str = Field(default="0.0.0.0", description="Host to bind the proxy server")
    port: int = Field(default=8000, description="Port to bind the proxy server")

    # Base configuration directory (contains models/, mcp/, skills/)
    config_dir: str = Field(
        default_factory=lambda: os.environ.get(
            "OPENAI_PROXY_CONFIG_DIR", _get_default_config_dir()
        ),
        description="Base directory for configuration files",
    )

    # Request timeout
    request_timeout: float = Field(
        default=300.0,
        description="Timeout for upstream API requests in seconds",
    )

    @computed_field
    @property
    def models_config_dir(self) -> str:
        """Directory containing model configuration files."""
        return str(Path(self.config_dir) / "models")

    @computed_field
    @property
    def mcp_config_dir(self) -> str:
        """Directory containing MCP server configuration."""
        return str(Path(self.config_dir) / "mcp")

    @computed_field
    @property
    def skills_dir(self) -> str:
        """Directory containing skills."""
        return str(Path(self.config_dir) / "skills")

    @computed_field
    @property
    def logs_dir(self) -> str:
        """Directory for log files."""
        return str(Path(self.config_dir) / "logs")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
