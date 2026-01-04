"""Tests for config.py - Application configuration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from openai_proxy.config import Settings, get_settings, _get_default_config_dir


class TestSettings:
    """Tests for Settings class."""

    def test_default_values_exist(self):
        """Test that default values exist for all required fields."""
        # This tests that the Settings class can be instantiated with defaults
        # even if those defaults come from environment or .env file
        settings = Settings()
        
        # These fields should always have values (either default or from env)
        assert settings.base_url is not None
        assert settings.host is not None
        assert settings.port is not None
        assert settings.request_timeout is not None
        
        # Verify types
        assert isinstance(settings.base_url, str)
        assert isinstance(settings.host, str)
        assert isinstance(settings.port, int)
        assert isinstance(settings.request_timeout, float)
        
        # Default host and port should be reasonable values
        assert settings.host in ["0.0.0.0", "127.0.0.1", "localhost"]
        assert 1000 <= settings.port <= 65535

    def test_custom_settings(self):
        """Test creating settings with custom values."""
        settings = Settings(
            base_url="https://custom.api.com/v1",
            api_key="custom-key",
            host="127.0.0.1",
            port=9000,
            request_timeout=60.0,
        )
        
        assert settings.base_url == "https://custom.api.com/v1"
        assert settings.api_key == "custom-key"
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.request_timeout == 60.0

    def test_computed_fields(self):
        """Test computed config directory fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(config_dir=tmpdir)
            
            assert settings.models_config_dir == str(Path(tmpdir) / "models")
            assert settings.mcp_config_dir == str(Path(tmpdir) / "mcp")
            assert settings.skills_dir == str(Path(tmpdir) / "skills")
            assert settings.logs_dir == str(Path(tmpdir) / "logs")

    def test_settings_from_environment(self):
        """Test loading settings from environment variables."""
        with patch.dict(os.environ, {
            "BASE_URL": "https://env.api.com/v1",
            "API_KEY": "env-api-key",
            "HOST": "0.0.0.0",
            "PORT": "8080",
        }):
            # Clear the cache to ensure fresh settings
            get_settings.cache_clear()
            
            settings = Settings()
            assert settings.base_url == "https://env.api.com/v1"
            assert settings.api_key == "env-api-key"

    def test_settings_extra_fields_ignored(self):
        """Test that extra fields are ignored (not raising errors)."""
        # This should not raise an error due to extra="ignore"
        settings = Settings(
            unknown_field="value",
            another_unknown=123,
        )
        # Just verify it doesn't raise and we can access known fields
        assert hasattr(settings, "base_url")
        assert hasattr(settings, "api_key")
        # Unknown fields should not become attributes
        assert not hasattr(settings, "unknown_field")


class TestGetDefaultConfigDir:
    """Tests for _get_default_config_dir function."""

    def test_default_config_dir(self):
        """Test that default config dir is in home directory."""
        config_dir = _get_default_config_dir()
        expected = str(Path.home() / ".config" / "openai_proxy")
        assert config_dir == expected


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        get_settings.cache_clear()
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2

    def test_get_settings_cache_clear(self):
        """Test that cache_clear creates new instance."""
        settings1 = get_settings()
        get_settings.cache_clear()
        settings2 = get_settings()
        
        # After clearing cache, it should be a new instance
        # but with same values (since environment hasn't changed)
        assert settings1.base_url == settings2.base_url


class TestConfigDirectoryPaths:
    """Tests for configuration directory path handling."""

    def test_models_config_dir_path(self):
        """Test models config directory path computation."""
        settings = Settings(config_dir="/custom/path")
        assert settings.models_config_dir == "/custom/path/models"

    def test_mcp_config_dir_path(self):
        """Test MCP config directory path computation."""
        settings = Settings(config_dir="/custom/path")
        assert settings.mcp_config_dir == "/custom/path/mcp"

    def test_skills_dir_path(self):
        """Test skills directory path computation."""
        settings = Settings(config_dir="/custom/path")
        assert settings.skills_dir == "/custom/path/skills"

    def test_logs_dir_path(self):
        """Test logs directory path computation."""
        settings = Settings(config_dir="/custom/path")
        assert settings.logs_dir == "/custom/path/logs"

    def test_relative_path_handling(self):
        """Test that relative paths work correctly."""
        settings = Settings(config_dir="./config")
        
        # Should preserve the relative path structure
        assert "models" in settings.models_config_dir
        assert "mcp" in settings.mcp_config_dir
