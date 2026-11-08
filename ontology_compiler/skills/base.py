"""Skill Base Classes — 技能抽象基类

R38-QA: 显式 Skill 抽象类定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional


class SkillPriority(IntEnum):
    """Skill execution priority."""
    CRITICAL = 0   # Must run first
    HIGH = 1       # Important optimization
    NORMAL = 2     # Standard processing
    LOW = 3        # Nice to have
    BACKGROUND = 4 # Post-processing


@dataclass
class SkillContext:
    """Context passed to skills during execution."""
    
    # Input data
    document_path: Optional[str] = None
    document_content: Optional[str] = None
    document_profile: Optional[Dict[str, Any]] = None
    
    # Extraction state
    extraction_result: Optional[Dict[str, Any]] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    quality_score: float = 0.0
    coverage: float = 0.0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    domain: str = "SUPPLY_CHAIN"
    
    # Runtime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from metadata."""
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in metadata."""
        self.metadata[key] = value


@dataclass
class SkillResult:
    """Result of skill execution."""
    
    success: bool
    skill_name: str
    
    # Output data
    output: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    execution_time_ms: int = 0
    cost_saved: float = 0.0  # Cost optimization metric
    
    # Status
    applied: bool = False  # Whether skill actually modified anything
    skipped: bool = False  # Whether skill was skipped
    reason: Optional[str] = None  # Reason for skip/failure
    
    # Error handling
    error: Optional[str] = None
    
    @classmethod
    def skipped_result(cls, skill_name: str, reason: str) -> "SkillResult":
        """Create a skipped result."""
        return cls(
            success=True,
            skill_name=skill_name,
            skipped=True,
            reason=reason
        )
    
    @classmethod
    def error_result(cls, skill_name: str, error: str) -> "SkillResult":
        """Create an error result."""
        return cls(
            success=False,
            skill_name=skill_name,
            error=error
        )


class Skill(ABC):
    """Base class for all skills.
    
    Skills are composable, reusable optimization or processing units
    that can be dynamically chained together.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.name = self._to_snake_case(self.__class__.__name__)
        self.config = config or {}
        self._enabled = True
    
    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert PascalCase to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        result = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        return result.replace('_skill', '')
    
    @property
    @abstractmethod
    def priority(self) -> SkillPriority:
        """Execution priority of this skill."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass
    
    @abstractmethod
    async def can_apply(self, context: SkillContext) -> bool:
        """Check if this skill can be applied to the given context.
        
        Args:
            context: Execution context
            
        Returns:
            True if skill should be applied
        """
        pass
    
    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """Execute the skill.
        
        Args:
            context: Execution context
            
        Returns:
            Skill execution result
        """
        pass
    
    def enable(self) -> None:
        """Enable this skill."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable this skill."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if skill is enabled."""
        return self._enabled
    
    def configure(self, **kwargs) -> "Skill":
        """Configure skill parameters.
        
        Returns:
            Self for chaining
        """
        self.config.update(kwargs)
        return self
