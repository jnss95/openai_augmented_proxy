"""Variable store for template processing.

This module provides a variable store that can load variables from different
sources (YAML files, databases, etc.) and make them available for use in
Jinja2 templates.

The store is designed to be extensible, allowing new backends to be added
easily (e.g., SQLite, PostgreSQL, Redis, etc.).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class VariableBackend(ABC):
    """Abstract base class for variable store backends."""

    @abstractmethod
    async def load(self) -> dict[str, Any]:
        """Load variables from the backend.
        
        Returns:
            Dictionary of variables
        """
        pass

    @abstractmethod
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a single variable by key.
        
        Args:
            key: The variable key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            The variable value or default
        """
        pass


class YamlVariableBackend(VariableBackend):
    """YAML file backend for variable storage.
    
    Loads variables from a YAML file. The file should contain a flat or
    nested dictionary structure.
    
    Example variables.yaml:
        app_name: "My Assistant"
        version: "1.0"
        settings:
            max_tokens: 4096
            temperature: 0.7
        prompts:
            greeting: "Hello! How can I help you today?"
    """

    def __init__(self, file_path: Path | str):
        """Initialize the YAML backend.
        
        Args:
            file_path: Path to the YAML variables file
        """
        self.file_path = Path(file_path)
        self._cache: dict[str, Any] | None = None

    async def load(self) -> dict[str, Any]:
        """Load all variables from the YAML file.
        
        Returns:
            Dictionary of all variables
        """
        if self._cache is not None:
            return self._cache

        if not self.file_path.exists():
            logger.debug(f"Variables file not found: {self.file_path}")
            self._cache = {}
            return self._cache

        try:
            with open(self.file_path, "r") as f:
                content = yaml.safe_load(f)
                self._cache = content if content else {}
                logger.info(f"Loaded {len(self._cache)} variables from {self.file_path}")
                return self._cache
        except yaml.YAMLError as e:
            logger.error(f"Error parsing variables file {self.file_path}: {e}")
            self._cache = {}
            return self._cache

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a variable by key, supporting dot notation.
        
        Args:
            key: Variable key (e.g., "settings.max_tokens")
            default: Default value if not found
            
        Returns:
            Variable value or default
        """
        data = await self.load()
        
        # Support dot notation for nested access
        parts = key.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current

    def invalidate_cache(self) -> None:
        """Invalidate the cache to force reload on next access."""
        self._cache = None


class VariableStore:
    """Main variable store that aggregates multiple backends.
    
    Variables can be loaded from multiple sources. When getting a variable,
    backends are searched in order of registration.
    
    Example usage:
        store = VariableStore()
        store.add_backend(YamlVariableBackend("/path/to/variables.yaml"))
        
        # Load all variables
        all_vars = await store.load_all()
        
        # Get a specific variable
        value = await store.get("app_name")
    """

    def __init__(self):
        """Initialize the variable store."""
        self._backends: list[VariableBackend] = []
        self._merged_cache: dict[str, Any] | None = None

    def add_backend(self, backend: VariableBackend) -> None:
        """Add a backend to the store.
        
        Backends added later have higher priority (override earlier ones).
        
        Args:
            backend: The backend to add
        """
        self._backends.append(backend)
        self._merged_cache = None

    async def load_all(self) -> dict[str, Any]:
        """Load and merge variables from all backends.
        
        Later backends override earlier ones for the same keys.
        
        Returns:
            Merged dictionary of all variables
        """
        if self._merged_cache is not None:
            return self._merged_cache

        merged: dict[str, Any] = {}
        
        for backend in self._backends:
            data = await backend.load()
            merged = self._deep_merge(merged, data)
        
        self._merged_cache = merged
        return merged

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a variable by key from the merged store.
        
        Args:
            key: Variable key (supports dot notation)
            default: Default value if not found
            
        Returns:
            Variable value or default
        """
        data = await self.load_all()
        
        # Support dot notation
        parts = key.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current

    def invalidate_cache(self) -> None:
        """Invalidate all caches."""
        self._merged_cache = None
        for backend in self._backends:
            if hasattr(backend, "invalidate_cache"):
                backend.invalidate_cache()

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Dictionary with values to override
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = VariableStore._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


# Global variable store instance
_variable_store: VariableStore | None = None


async def get_variable_store() -> VariableStore:
    """Get the global variable store instance.
    
    Returns:
        The global VariableStore instance
    """
    global _variable_store
    if _variable_store is None:
        _variable_store = VariableStore()
    return _variable_store


def set_variable_store(store: VariableStore) -> None:
    """Set the global variable store instance.
    
    Args:
        store: The VariableStore to set as global
    """
    global _variable_store
    _variable_store = store


async def initialize_variable_store(config_dir: Path | str) -> VariableStore:
    """Initialize the variable store with default backends.
    
    This sets up the YAML backend looking for variables.yaml in the config dir.
    
    Args:
        config_dir: Path to the configuration directory
        
    Returns:
        Initialized VariableStore
    """
    config_path = Path(config_dir)
    variables_file = config_path / "variables.yaml"
    
    store = VariableStore()
    store.add_backend(YamlVariableBackend(variables_file))
    
    # Pre-load to validate and log
    await store.load_all()
    
    set_variable_store(store)
    return store
