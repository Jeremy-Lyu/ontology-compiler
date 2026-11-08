"""Skill bounded-context port definitions.

Ports and DTOs for skill discovery, validation and execution. Split
from `all_ports.py` during P2 port classification (§9.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SkillManifest:
    """Skill metadata from skill.json."""
    name: str
    description: str
    category: str
    enabled: bool
    script_path: str
    args_schema: dict[str, Any]


@dataclass
class SkillExecutionResult:
    """Result of skill execution."""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time_ms: Optional[int] = None


class SkillRegistryPort(ABC):
    """Port interface for skill registry operations."""

    @abstractmethod
    def discover_skills(self, skill_root: str) -> list[SkillManifest]:
        """Discover all available skills from the given root directory."""
        pass

    @abstractmethod
    def get_skill(self, name: str) -> Optional[SkillManifest]:
        """Get a specific skill by name."""
        pass

    @abstractmethod
    def list_skills(
        self,
        category: Optional[str] = None,
        enabled_only: bool = True
    ) -> list[SkillManifest]:
        """List skills with optional filtering."""
        pass

    @abstractmethod
    def execute_skill(
        self,
        name: str,
        args: dict[str, Any],
        working_dir: Optional[str] = None
    ) -> SkillExecutionResult:
        """Execute a skill with given arguments."""
        pass

    @abstractmethod
    def validate_skill_config(self, name: str) -> tuple[bool, list[str]]:
        """Validate a skill's configuration."""
        pass

    @abstractmethod
    def get_skill_categories(self) -> list[str]:
        """Get all available skill categories."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        pass


__all__ = [
    "SkillManifest",
    "SkillExecutionResult",
    "SkillRegistryPort",
]
