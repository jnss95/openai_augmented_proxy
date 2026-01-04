"""Tests for skills.py - Skills loader."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from openai_proxy.skills import (
    Skill,
    SkillsLoader,
    get_skills_loader,
    reload_skills_loader,
)


class TestSkill:
    """Tests for Skill model."""

    def test_skill_creation(self):
        """Test creating a skill."""
        skill = Skill(
            name="test-skill",
            description="A test skill",
            content="# Test\n\nThis is test content.",
            source_file="/path/to/skill.md"
        )
        
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.content == "# Test\n\nThis is test content."
        assert skill.source_file == "/path/to/skill.md"

    def test_skill_minimal(self):
        """Test creating skill with minimal fields."""
        skill = Skill(name="minimal", content="Content")
        
        assert skill.name == "minimal"
        assert skill.description == ""
        assert skill.source_file == ""


class TestSkillsLoader:
    """Tests for SkillsLoader."""

    @pytest.fixture
    def skills_dir(self) -> Path:
        """Create a temporary skills directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def loader(self, skills_dir) -> SkillsLoader:
        """Create a SkillsLoader for the temp directory."""
        return SkillsLoader(skills_dir)

    def test_load_all_empty_directory(self, loader, skills_dir):
        """Test loading from empty directory."""
        loader.load_all()
        
        assert loader.get_global_skills() == []

    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        loader = SkillsLoader("/nonexistent/path")
        loader.load_all()
        
        assert loader.get_global_skills() == []

    def test_load_markdown_skill(self, loader, skills_dir):
        """Test loading a markdown skill file."""
        skill_content = """# Code Review Expert

You are an expert at reviewing code.

## Guidelines

1. Check for bugs
2. Suggest improvements
"""
        (skills_dir / "code-review.md").write_text(skill_content)
        
        loader.load_all()
        
        skills = loader.get_global_skills()
        assert len(skills) == 1
        assert skills[0].name == "code-review"
        assert skills[0].description == "Code Review Expert"
        assert "Check for bugs" in skills[0].content

    def test_load_yaml_skill(self, loader, skills_dir):
        """Test loading a YAML skill file."""
        skill_data = {
            "name": "yaml-skill",
            "description": "A YAML-defined skill",
            "content": "This is the skill content."
        }
        
        with open(skills_dir / "yaml-skill.yaml", "w") as f:
            yaml.dump(skill_data, f)
        
        loader.load_all()
        
        skills = loader.get_global_skills()
        assert len(skills) == 1
        assert skills[0].name == "yaml-skill"
        assert skills[0].description == "A YAML-defined skill"

    def test_load_yml_skill(self, loader, skills_dir):
        """Test loading a .yml skill file."""
        skill_data = {
            "name": "yml-skill",
            "content": "Content here"
        }
        
        with open(skills_dir / "skill.yml", "w") as f:
            yaml.dump(skill_data, f)
        
        loader.load_all()
        
        skill = loader.get_skill("yml-skill")
        assert skill is not None

    def test_load_multiple_skills(self, loader, skills_dir):
        """Test loading multiple skill files."""
        (skills_dir / "skill1.md").write_text("# Skill 1\n\nContent 1")
        (skills_dir / "skill2.md").write_text("# Skill 2\n\nContent 2")
        
        loader.load_all()
        
        skills = loader.get_global_skills()
        assert len(skills) == 2

    def test_load_model_specific_skills(self, loader, skills_dir):
        """Test loading model-specific skills from subdirectory."""
        # Create model-specific skill directory
        model_dir = skills_dir / "my-model"
        model_dir.mkdir()
        (model_dir / "specific.md").write_text("# Model Specific\n\nSpecific content")
        
        loader.load_all()
        
        # Global skills should be empty
        assert loader.get_global_skills() == []
        
        # Model-specific skills should be loaded
        model_skills = loader.get_model_skills("my-model")
        assert len(model_skills) == 1
        assert model_skills[0].name == "specific"

    def test_get_skill_from_global(self, loader, skills_dir):
        """Test getting a specific global skill."""
        (skills_dir / "test.md").write_text("# Test\n\nContent")
        
        loader.load_all()
        
        skill = loader.get_skill("test")
        assert skill is not None
        assert skill.name == "test"

    def test_get_skill_from_model(self, loader, skills_dir):
        """Test getting a skill from model-specific directory."""
        model_dir = skills_dir / "model1"
        model_dir.mkdir()
        (model_dir / "specific.md").write_text("# Specific\n\nContent")
        
        loader.load_all()
        
        # Get with model name
        skill = loader.get_skill("specific", model_name="model1")
        assert skill is not None
        
        # Without model name, should return None (not in global)
        skill = loader.get_skill("specific")
        assert skill is None

    def test_get_skill_nonexistent(self, loader, skills_dir):
        """Test getting non-existent skill returns None."""
        loader.load_all()
        
        skill = loader.get_skill("nonexistent")
        assert skill is None


class TestSkillsLoaderFiltering:
    """Tests for skill filtering by model."""

    @pytest.fixture
    def populated_loader(self) -> SkillsLoader:
        """Create a loader with multiple skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            
            # Global skills
            (skills_dir / "global1.md").write_text("# Global 1\n\nContent")
            (skills_dir / "global2.md").write_text("# Global 2\n\nContent")
            
            # Model-specific skills
            model_dir = skills_dir / "test-model"
            model_dir.mkdir()
            (model_dir / "specific.md").write_text("# Specific\n\nContent")
            
            loader = SkillsLoader(skills_dir)
            loader.load_all()
            
            yield loader

    def test_get_skills_for_model_all(self, populated_loader):
        """Test getting all skills for a model."""
        skills = populated_loader.get_skills_for_model(
            model_name="test-model",
            include_global=True
        )
        
        # Should include both global and model-specific
        names = [s.name for s in skills]
        assert "global1" in names
        assert "global2" in names
        assert "specific" in names

    def test_get_skills_for_model_no_global(self, populated_loader):
        """Test getting only model-specific skills."""
        skills = populated_loader.get_skills_for_model(
            model_name="test-model",
            include_global=False
        )
        
        # Should only include model-specific
        names = [s.name for s in skills]
        assert "specific" in names
        assert "global1" not in names

    def test_get_skills_for_model_filter_by_name(self, populated_loader):
        """Test filtering skills by name."""
        skills = populated_loader.get_skills_for_model(
            model_name="test-model",
            skill_names=["global1", "specific"],
            include_global=True
        )
        
        names = [s.name for s in skills]
        assert "global1" in names
        assert "specific" in names
        assert "global2" not in names

    def test_get_skills_for_unknown_model(self, populated_loader):
        """Test getting skills for unknown model."""
        skills = populated_loader.get_skills_for_model(
            model_name="unknown-model",
            include_global=True
        )
        
        # Should still get global skills
        names = [s.name for s in skills]
        assert "global1" in names
        assert "global2" in names


class TestSkillsLoaderFormatting:
    """Tests for skill formatting."""

    @pytest.fixture
    def loader_with_skills(self) -> SkillsLoader:
        """Create a loader with test skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            (skills_dir / "skill1.md").write_text("Skill 1 content")
            (skills_dir / "skill2.md").write_text("Skill 2 content")
            
            loader = SkillsLoader(skills_dir)
            loader.load_all()
            yield loader

    def test_format_skills_for_prompt(self, loader_with_skills):
        """Test formatting skills into a prompt."""
        skills = loader_with_skills.get_global_skills()
        formatted = loader_with_skills.format_skills_for_prompt(skills)
        
        assert "## Skill: skill1" in formatted
        assert "## Skill: skill2" in formatted
        assert "Skill 1 content" in formatted
        assert "Skill 2 content" in formatted

    def test_format_skills_empty(self, loader_with_skills):
        """Test formatting empty skills list."""
        formatted = loader_with_skills.format_skills_for_prompt([])
        assert formatted == ""

    def test_format_skills_custom_separator(self, loader_with_skills):
        """Test formatting with custom separator."""
        skills = loader_with_skills.get_global_skills()
        formatted = loader_with_skills.format_skills_for_prompt(skills, separator="\n===\n")
        
        assert "\n===\n" in formatted


class TestGlobalSkillsLoader:
    """Tests for global skills loader functions."""

    def test_get_skills_loader_creates_singleton(self):
        """Test that get_skills_loader creates a singleton."""
        import openai_proxy.skills as skills_module
        skills_module._skills_loader = None
        
        with patch("openai_proxy.skills.get_settings") as mock_settings:
            mock_settings.return_value.skills_dir = "/nonexistent"
            
            loader1 = get_skills_loader()
            loader2 = get_skills_loader()
            
            assert loader1 is loader2

    def test_reload_skills_loader(self):
        """Test that reload creates a new loader."""
        import openai_proxy.skills as skills_module
        skills_module._skills_loader = None
        
        with patch("openai_proxy.skills.get_settings") as mock_settings:
            mock_settings.return_value.skills_dir = "/nonexistent"
            
            loader1 = get_skills_loader()
            loader2 = reload_skills_loader()
            
            assert loader1 is not loader2


class TestYAMLSkillWithContentFile:
    """Tests for YAML skills with external content files."""

    def test_yaml_skill_with_content_file(self):
        """Test loading YAML skill that references external content file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            
            # Create content file
            (skills_dir / "content.md").write_text("External content here")
            
            # Create YAML skill referencing it
            skill_data = {
                "name": "external-skill",
                "description": "Skill with external content",
                "content_file": "content.md"
            }
            
            with open(skills_dir / "skill.yaml", "w") as f:
                yaml.dump(skill_data, f)
            
            loader = SkillsLoader(skills_dir)
            loader.load_all()
            
            skill = loader.get_skill("external-skill")
            assert skill is not None
            assert skill.content == "External content here"
