"""Skills package initialization.

Compatibility imports for legacy direct ``skills.*`` call sites.
Canonical skill discovery is ``ontology_compiler.harness.skill_registry``.

``skills.skill_registry`` was archived 2026-11-08 — use harness /
``SkillsRouter`` (harness-backed) instead.
"""

from .skill_memory import skill_memory, SkillMemory
from .skill_metadata import SkillMetadata
from .skill_router import SkillsRouter
from .skill_interface import BaseSkill
from .skill_contract import UnifiedSkillContract, UnifiedSkillContractValidator

skills_router = SkillsRouter()

__all__ = [
    "skill_memory",
    "SkillMemory",
    "SkillMetadata",
    "SkillsRouter",
    "skills_router",
    "BaseSkill",
    "UnifiedSkillContract",
    "UnifiedSkillContractValidator",
]
