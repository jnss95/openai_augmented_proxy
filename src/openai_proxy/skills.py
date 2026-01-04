"""Skills loader for Claude Code-style skills.

Skills are markdown files that contain instructions, context, or capabilities
that can be injected into the system prompt for specific models.

Skills can be:
- Global skills (in conf/skills/) available to all models
- Model-specific skills (in conf/skills/{model_name}/)
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .config import get_settings


class Skill(BaseModel):
    """A skill that can be attached to a model."""

    name: str
    description: str = ""
    content: str  # The markdown content of the skill
    source_file: str = ""  # Path to the source file


class SkillsLoader:
    """Loader for skills from the filesystem."""

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self._global_skills: dict[str, Skill] = {}
        self._model_skills: dict[str, dict[str, Skill]] = {}  # model_name -> {skill_name -> Skill}

    def load_all(self) -> None:
        """Load all skills from the skills directory."""
        if not self.skills_dir.exists():
            return

        # Load global skills (directly in skills_dir)
        self._load_skills_from_dir(self.skills_dir, self._global_skills)

        # Load model-specific skills (in subdirectories)
        for subdir in self.skills_dir.iterdir():
            if subdir.is_dir():
                model_name = subdir.name
                self._model_skills[model_name] = {}
                self._load_skills_from_dir(subdir, self._model_skills[model_name])

    def _load_skills_from_dir(self, directory: Path, target: dict[str, Skill]) -> None:
        """Load skills from a directory into the target dict."""
        for file_path in directory.glob("*.md"):
            try:
                skill = self._load_skill_file(file_path)
                if skill:
                    target[skill.name] = skill
            except Exception as e:
                print(f"Warning: Failed to load skill from {file_path}: {e}")

        # Also support YAML skill definitions
        for file_path in directory.glob("*.yaml"):
            try:
                skill = self._load_skill_yaml(file_path)
                if skill:
                    target[skill.name] = skill
            except Exception as e:
                print(f"Warning: Failed to load skill from {file_path}: {e}")

        for file_path in directory.glob("*.yml"):
            try:
                skill = self._load_skill_yaml(file_path)
                if skill:
                    target[skill.name] = skill
            except Exception as e:
                print(f"Warning: Failed to load skill from {file_path}: {e}")

    def _load_skill_file(self, file_path: Path) -> Skill | None:
        """Load a skill from a markdown file."""
        content = file_path.read_text(encoding="utf-8")
        
        # Extract name from filename (without extension)
        name = file_path.stem
        
        # Try to extract description from the first line if it's a header
        description = ""
        lines = content.strip().split("\n")
        if lines and lines[0].startswith("# "):
            description = lines[0][2:].strip()

        return Skill(
            name=name,
            description=description,
            content=content,
            source_file=str(file_path),
        )

    def _load_skill_yaml(self, file_path: Path) -> Skill | None:
        """Load a skill from a YAML file."""
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return None

        name = data.get("name", file_path.stem)
        description = data.get("description", "")
        content = data.get("content", "")

        # If content_file is specified, load content from that file
        if "content_file" in data:
            content_path = file_path.parent / data["content_file"]
            if content_path.exists():
                content = content_path.read_text(encoding="utf-8")

        return Skill(
            name=name,
            description=description,
            content=content,
            source_file=str(file_path),
        )

    def get_global_skills(self) -> list[Skill]:
        """Get all global skills."""
        return list(self._global_skills.values())

    def get_model_skills(self, model_name: str) -> list[Skill]:
        """Get skills for a specific model."""
        return list(self._model_skills.get(model_name, {}).values())

    def get_skills_for_model(
        self,
        model_name: str,
        skill_names: list[str] | None = None,
        include_global: bool = True,
    ) -> list[Skill]:
        """Get skills for a model, optionally filtering by name."""
        skills: list[Skill] = []

        # Add global skills if requested
        if include_global:
            if skill_names:
                skills.extend(
                    s for s in self._global_skills.values()
                    if s.name in skill_names
                )
            else:
                skills.extend(self._global_skills.values())

        # Add model-specific skills
        model_skills = self._model_skills.get(model_name, {})
        if skill_names:
            skills.extend(
                s for s in model_skills.values()
                if s.name in skill_names
            )
        else:
            skills.extend(model_skills.values())

        return skills

    def get_skill(self, name: str, model_name: str | None = None) -> Skill | None:
        """Get a specific skill by name."""
        # Check model-specific skills first
        if model_name and model_name in self._model_skills:
            if name in self._model_skills[model_name]:
                return self._model_skills[model_name][name]

        # Fall back to global skills
        return self._global_skills.get(name)

    def format_skills_for_prompt(
        self,
        skills: list[Skill],
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Format skills into a string suitable for injection into a prompt."""
        if not skills:
            return ""

        formatted_parts = []
        for skill in skills:
            formatted_parts.append(f"## Skill: {skill.name}\n\n{skill.content}")

        return separator.join(formatted_parts)


# Global skills loader instance
_skills_loader: SkillsLoader | None = None


def get_skills_loader() -> SkillsLoader:
    """Get the global skills loader, initializing if needed."""
    global _skills_loader
    if _skills_loader is None:
        settings = get_settings()
        skills_dir = Path(settings.skills_dir)
        _skills_loader = SkillsLoader(skills_dir)
        _skills_loader.load_all()
    return _skills_loader


def reload_skills_loader() -> SkillsLoader:
    """Force reload of the skills loader."""
    global _skills_loader
    _skills_loader = None
    return get_skills_loader()
