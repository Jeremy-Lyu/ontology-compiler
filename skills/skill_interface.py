"""Base interface for all RAG Skills.

[DEPRECATED] This module is kept for backward compatibility.
The canonical Skill base class lives in `ontology_compiler.skills.base`.
New skills should inherit from `Skill` directly.

This module provides a `BaseSkill` adapter that wraps the canonical `Skill`
interface for legacy code that still imports from `skills.skill_interface`.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

try:
    from ontology_compiler.skills.base import Skill, SkillContext, SkillResult
    _CANONICAL_AVAILABLE = True
except ImportError:
    _CANONICAL_AVAILABLE = False
    Skill = None
    SkillContext = None
    SkillResult = None


class BaseSkill(ABC):
    """Abstract base class for a Skill (legacy adapter).

    New code should use `ontology_compiler.skills.base.Skill` directly.
    This class provides backward compatibility for existing skill scripts.

    Attributes
    ----------
    name: str
        Human readable name of the skill.
    description: str
        Short description of what the skill does.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the skill (synchronous, legacy interface).

        Parameters
        ----------
        context: dict
            Arbitrary data provided by the orchestrator.
        """
        raise NotImplementedError

    async def execute_async(self, context: Any) -> Any:
        """Async execution adapter for compatibility with canonical Skill.

        Wraps the synchronous `execute` method.
        """
        if isinstance(context, SkillContext):
            ctx_dict = context.to_dict() if hasattr(context, 'to_dict') else context.__dict__
        else:
            ctx_dict = context
        result = self.execute(ctx_dict)
        if SkillResult and isinstance(result, SkillResult):
            return result
        if SkillResult:
            return SkillResult(success=True, data=result, skill_name=self.name)
        return result
