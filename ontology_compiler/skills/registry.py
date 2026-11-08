"""Skill Registry — 技能注册与管理

R38-QA: Centralized skill discovery and management.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from .base import Skill, SkillPriority

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Registry for managing and discovering skills.
    
    Provides centralized skill management with:
    - Skill registration and discovery
    - Priority-based ordering
    - Configuration management
    - Enable/disable control
    """
    
    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._skill_metadata: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, List[str]] = {
            "optimization": [],
            "extraction": [],
            "validation": [],
            "post_processing": [],
        }
        logger.info("SkillRegistry initialized")
    
    def register(self, skill: Skill, category: Optional[str] = None) -> None:
        """Register a skill.
        
        Args:
            skill: Skill instance to register
            category: Optional category for grouping
        """
        name = skill.name
        
        if name in self._skills:
            logger.warning(f"Skill '{name}' already registered, overwriting")
        
        self._skills[name] = skill
        
        if category and category in self._categories:
            self._categories[category].append(name)
        
        logger.info(f"Registered skill: {name} (priority={skill.priority.name})")
    
    def unregister(self, name: str) -> bool:
        """Unregister a skill.
        
        Args:
            name: Skill name
            
        Returns:
            True if skill was found and removed
        """
        if name in self._skills:
            del self._skills[name]
            
            # Remove from categories
            for cat_list in self._categories.values():
                if name in cat_list:
                    cat_list.remove(name)
            
            logger.info(f"Unregistered skill: {name}")
            return True
        
        return False
    
    def get(self, name: str) -> Optional[Skill]:
        """Get skill by name.
        
        Args:
            name: Skill name
            
        Returns:
            Skill instance or None
        """
        return self._skills.get(name)

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata-only skill description by name."""
        return self._skill_metadata.get(name)
    
    def list_skills(self, enabled_only: bool = False) -> List[str]:
        """List all registered skill names.

        Args:
            enabled_only: Only return enabled skills

        Returns:
            List of skill names
        """
        if enabled_only:
            return [name for name, skill in self._skills.items() if skill.is_enabled()]
        return list(self._skills.keys())

    def list_active(self) -> List[str]:
        """Return all active (enabled) skill names.

        Legacy compatibility alias for list_skills(enabled_only=True).
        Also includes skills loaded from metadata (skill.json) that have
        no runtime instance but are considered active.
        """
        active = [name for name, skill in self._skills.items() if skill.is_enabled()]
        # Include metadata-only skills (loaded from skill.json) as active
        for name in self._skill_metadata:
            if name not in active:
                active.append(name)
        return active
    
    def get_by_priority(self, priority: SkillPriority) -> List[Skill]:
        """Get all skills with given priority.
        
        Args:
            priority: Priority level
            
        Returns:
            List of skills
        """
        return [
            skill for skill in self._skills.values()
            if skill.priority == priority and skill.is_enabled()
        ]
    
    def get_all_ordered(self) -> List[Skill]:
        """Get all enabled skills ordered by priority.
        
        Returns:
            List of skills sorted by priority
        """
        enabled = [s for s in self._skills.values() if s.is_enabled()]
        return sorted(enabled, key=lambda s: s.priority.value)
    
    def enable(self, name: str) -> bool:
        """Enable a skill.
        
        Args:
            name: Skill name
            
        Returns:
            True if skill was found and enabled
        """
        skill = self._skills.get(name)
        if skill:
            skill.enable()
            logger.info(f"Enabled skill: {name}")
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """Disable a skill.
        
        Args:
            name: Skill name
            
        Returns:
            True if skill was found and disabled
        """
        skill = self._skills.get(name)
        if skill:
            skill.disable()
            logger.info(f"Disabled skill: {name}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with statistics
        """
        total = len(self._skills)
        enabled = sum(1 for s in self._skills.values() if s.is_enabled())
        
        by_priority = {}
        for p in SkillPriority:
            count = len(self.get_by_priority(p))
            if count > 0:
                by_priority[p.name] = count
        
        return {
            "total": total,
            "enabled": enabled,
            "disabled": total - enabled,
            "by_priority": by_priority,
            "by_category": {k: len(v) for k, v in self._categories.items()},
        }
    
    def clear(self) -> None:
        """Clear all registered skills."""
        self._skills.clear()
        for cat_list in self._categories.values():
            cat_list.clear()
        logger.info("SkillRegistry cleared")

    def load_skills(self, skills_base_path: Optional[str] = None) -> int:
        """Compatibility alias for loading skill metadata from the legacy skills tree.

        The canonical runtime registry keeps metadata-only skills so legacy discovery
        code can still work while routing all runtime state through one registry.
        """
        if skills_base_path is None:
            import os
            skills_base_path = os.path.join(os.getcwd(), "skills")
        return self.load_metadata_from_skill_json(skills_base_path)

    def load_metadata_from_skill_json(self, skills_base_path: str) -> int:
        """Load skill metadata from skills/*/skill.json files.

        This bridges the legacy metadata registry (skills/skill_registry.py)
        into the canonical runtime registry so there is a single entry point.

        Args:
            skills_base_path: Path to the skills/ directory

        Returns:
            Number of skill metadata entries loaded
        """
        import json
        import glob

        patterns = [
            f"{skills_base_path}/*/.ontology-skill/skill.json",
            f"{skills_base_path}/*/*/.ontology-skill/skill.json",
            f"{skills_base_path}/*/*/skill.json",
            f"{skills_base_path}/*/skill.json",
        ]

        loaded = 0
        for pattern in patterns:
            for skill_json_path in sorted(glob.glob(pattern)):
                try:
                    with open(skill_json_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    skill_name = meta.get("name", "")
                    if not skill_name:
                        continue
                    # Store metadata alongside any registered instance
                    if skill_name not in self._skills:
                        # Create a lightweight metadata entry
                        self._skill_metadata[skill_name] = meta
                    loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load skill metadata from {skill_json_path}: {e}")

        logger.info(f"Loaded {loaded} skill metadata entries from {skills_base_path}")
        return loaded

    def get_metadata(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get skill metadata (from skill.json files)."""
        return self._skill_metadata.get(skill_name)

    def list_all_metadata(self) -> Dict[str, Any]:
        """List all skill metadata entries."""
        return dict(self._skill_metadata)


# Global registry instance
_global_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get global skill registry (singleton).
    
    Returns:
        Global SkillRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def reset_skill_registry() -> None:
    """Reset global registry (for testing)."""
    global _global_registry
    _global_registry = None
