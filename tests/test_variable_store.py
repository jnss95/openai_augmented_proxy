"""Tests for variable_store.py - Variable storage and retrieval."""

from pathlib import Path

import pytest

from openai_proxy.variable_store import (
    VariableBackend,
    VariableStore,
    YamlVariableBackend,
    get_variable_store,
    initialize_variable_store,
    set_variable_store,
)


class TestYamlVariableBackend:
    """Tests for YamlVariableBackend class."""

    @pytest.fixture
    def vars_file(self, tmp_path):
        """Create a temporary variables file."""
        file_path = tmp_path / "variables.yaml"
        file_path.write_text("""
app_name: "Test Application"
version: "2.0"
count: 42
enabled: true
settings:
  max_tokens: 4096
  temperature: 0.7
  nested:
    deep_value: "found"
items:
  - first
  - second
  - third
""")
        return file_path

    @pytest.mark.asyncio
    async def test_load_variables(self, vars_file):
        """Test loading variables from YAML file."""
        backend = YamlVariableBackend(vars_file)
        data = await backend.load()
        
        assert data["app_name"] == "Test Application"
        assert data["version"] == "2.0"
        assert data["count"] == 42
        assert data["enabled"] is True

    @pytest.mark.asyncio
    async def test_load_nested_variables(self, vars_file):
        """Test loading nested variables."""
        backend = YamlVariableBackend(vars_file)
        data = await backend.load()
        
        assert data["settings"]["max_tokens"] == 4096
        assert data["settings"]["temperature"] == 0.7
        assert data["settings"]["nested"]["deep_value"] == "found"

    @pytest.mark.asyncio
    async def test_load_list_variables(self, vars_file):
        """Test loading list variables."""
        backend = YamlVariableBackend(vars_file)
        data = await backend.load()
        
        assert data["items"] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file."""
        backend = YamlVariableBackend(tmp_path / "nonexistent.yaml")
        data = await backend.load()
        
        assert data == {}

    @pytest.mark.asyncio
    async def test_load_empty_file(self, tmp_path):
        """Test loading from empty file."""
        file_path = tmp_path / "empty.yaml"
        file_path.write_text("")
        
        backend = YamlVariableBackend(file_path)
        data = await backend.load()
        
        assert data == {}

    @pytest.mark.asyncio
    async def test_load_invalid_yaml(self, tmp_path):
        """Test loading from invalid YAML file."""
        file_path = tmp_path / "invalid.yaml"
        file_path.write_text("invalid: yaml: content: [")
        
        backend = YamlVariableBackend(file_path)
        data = await backend.load()
        
        assert data == {}

    @pytest.mark.asyncio
    async def test_get_simple_key(self, vars_file):
        """Test getting a simple key."""
        backend = YamlVariableBackend(vars_file)
        
        value = await backend.get("app_name")
        assert value == "Test Application"

    @pytest.mark.asyncio
    async def test_get_nested_key_dot_notation(self, vars_file):
        """Test getting nested key with dot notation."""
        backend = YamlVariableBackend(vars_file)
        
        value = await backend.get("settings.max_tokens")
        assert value == 4096

    @pytest.mark.asyncio
    async def test_get_deep_nested_key(self, vars_file):
        """Test getting deeply nested key."""
        backend = YamlVariableBackend(vars_file)
        
        value = await backend.get("settings.nested.deep_value")
        assert value == "found"

    @pytest.mark.asyncio
    async def test_get_missing_key_returns_default(self, vars_file):
        """Test getting missing key returns default."""
        backend = YamlVariableBackend(vars_file)
        
        value = await backend.get("nonexistent", "default_value")
        assert value == "default_value"

    @pytest.mark.asyncio
    async def test_get_missing_nested_key_returns_default(self, vars_file):
        """Test getting missing nested key returns default."""
        backend = YamlVariableBackend(vars_file)
        
        value = await backend.get("settings.nonexistent", "fallback")
        assert value == "fallback"

    @pytest.mark.asyncio
    async def test_caching(self, vars_file):
        """Test that data is cached after first load."""
        backend = YamlVariableBackend(vars_file)
        
        # First load
        data1 = await backend.load()
        
        # Modify file (shouldn't affect cached data)
        vars_file.write_text("app_name: Modified")
        
        # Second load should return cached data
        data2 = await backend.load()
        
        assert data1 == data2
        assert data2["app_name"] == "Test Application"

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, vars_file):
        """Test cache invalidation."""
        backend = YamlVariableBackend(vars_file)
        
        # First load
        await backend.load()
        
        # Modify file
        vars_file.write_text("app_name: Modified")
        
        # Invalidate cache
        backend.invalidate_cache()
        
        # Reload should get new data
        data = await backend.load()
        assert data["app_name"] == "Modified"


class TestVariableStore:
    """Tests for VariableStore class."""

    @pytest.fixture
    def store(self):
        """Create an empty variable store."""
        return VariableStore()

    @pytest.fixture
    def yaml_backend(self, tmp_path):
        """Create a YAML backend with test data."""
        file_path = tmp_path / "vars.yaml"
        file_path.write_text("""
app: "TestApp"
config:
  timeout: 30
""")
        return YamlVariableBackend(file_path)

    @pytest.mark.asyncio
    async def test_empty_store(self, store):
        """Test empty store returns empty dict."""
        data = await store.load_all()
        assert data == {}

    @pytest.mark.asyncio
    async def test_add_backend(self, store, yaml_backend):
        """Test adding a backend."""
        store.add_backend(yaml_backend)
        data = await store.load_all()
        
        assert data["app"] == "TestApp"
        assert data["config"]["timeout"] == 30

    @pytest.mark.asyncio
    async def test_multiple_backends_merge(self, tmp_path):
        """Test that multiple backends are merged."""
        # First backend
        file1 = tmp_path / "vars1.yaml"
        file1.write_text("""
shared: from_first
only_first: value1
""")
        
        # Second backend
        file2 = tmp_path / "vars2.yaml"
        file2.write_text("""
shared: from_second
only_second: value2
""")
        
        store = VariableStore()
        store.add_backend(YamlVariableBackend(file1))
        store.add_backend(YamlVariableBackend(file2))
        
        data = await store.load_all()
        
        # Later backends override
        assert data["shared"] == "from_second"
        # Both unique values present
        assert data["only_first"] == "value1"
        assert data["only_second"] == "value2"

    @pytest.mark.asyncio
    async def test_deep_merge(self, tmp_path):
        """Test deep merging of nested structures."""
        file1 = tmp_path / "vars1.yaml"
        file1.write_text("""
config:
  timeout: 30
  retries: 3
""")
        
        file2 = tmp_path / "vars2.yaml"
        file2.write_text("""
config:
  timeout: 60
  debug: true
""")
        
        store = VariableStore()
        store.add_backend(YamlVariableBackend(file1))
        store.add_backend(YamlVariableBackend(file2))
        
        data = await store.load_all()
        
        # Merged config
        assert data["config"]["timeout"] == 60  # Overridden
        assert data["config"]["retries"] == 3   # From first
        assert data["config"]["debug"] is True  # From second

    @pytest.mark.asyncio
    async def test_get_simple_value(self, store, yaml_backend):
        """Test getting a simple value."""
        store.add_backend(yaml_backend)
        
        value = await store.get("app")
        assert value == "TestApp"

    @pytest.mark.asyncio
    async def test_get_nested_value(self, store, yaml_backend):
        """Test getting a nested value."""
        store.add_backend(yaml_backend)
        
        value = await store.get("config.timeout")
        assert value == 30

    @pytest.mark.asyncio
    async def test_get_missing_returns_default(self, store):
        """Test getting missing key returns default."""
        value = await store.get("missing", "default")
        assert value == "default"

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, tmp_path):
        """Test cache invalidation."""
        file_path = tmp_path / "vars.yaml"
        file_path.write_text("value: original")
        
        store = VariableStore()
        store.add_backend(YamlVariableBackend(file_path))
        
        # Load initial
        await store.load_all()
        
        # Modify file
        file_path.write_text("value: modified")
        
        # Invalidate and reload
        store.invalidate_cache()
        data = await store.load_all()
        
        assert data["value"] == "modified"


class TestGlobalFunctions:
    """Tests for global variable store functions."""

    @pytest.mark.asyncio
    async def test_get_variable_store_creates_instance(self):
        """Test that get_variable_store creates an instance."""
        store = await get_variable_store()
        assert store is not None
        assert isinstance(store, VariableStore)

    def test_set_variable_store(self):
        """Test setting the global store."""
        custom_store = VariableStore()
        set_variable_store(custom_store)
        
        # We need to access the global directly to verify
        from openai_proxy import variable_store
        assert variable_store._variable_store is custom_store

    @pytest.mark.asyncio
    async def test_initialize_variable_store(self, tmp_path):
        """Test initializing the variable store."""
        # Create a variables file
        vars_file = tmp_path / "variables.yaml"
        vars_file.write_text("test_key: test_value")
        
        store = await initialize_variable_store(tmp_path)
        
        assert store is not None
        value = await store.get("test_key")
        assert value == "test_value"

    @pytest.mark.asyncio
    async def test_initialize_without_vars_file(self, tmp_path):
        """Test initializing without a variables file."""
        # Don't create the file
        store = await initialize_variable_store(tmp_path)
        
        assert store is not None
        data = await store.load_all()
        assert data == {}


class TestDeepMerge:
    """Tests for the deep merge utility."""

    def test_simple_merge(self):
        """Test simple merge of flat dicts."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        
        result = VariableStore._deep_merge(base, override)
        
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test merge of nested dicts."""
        base = {"outer": {"inner1": 1, "inner2": 2}}
        override = {"outer": {"inner2": 20, "inner3": 3}}
        
        result = VariableStore._deep_merge(base, override)
        
        assert result == {"outer": {"inner1": 1, "inner2": 20, "inner3": 3}}

    def test_non_dict_override(self):
        """Test that non-dict values override entirely."""
        base = {"key": {"nested": "value"}}
        override = {"key": "simple"}
        
        result = VariableStore._deep_merge(base, override)
        
        assert result == {"key": "simple"}

    def test_base_not_modified(self):
        """Test that base dict is not modified."""
        base = {"a": 1}
        override = {"b": 2}
        
        VariableStore._deep_merge(base, override)
        
        assert base == {"a": 1}
