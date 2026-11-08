"""Base classes, protocols, and shared data structures for quality assurance.

This module consolidates shared types from conflict_engine.py and quality_gate_rules.py.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Conflict Detection Types (from conflict_engine.py)
# =============================================================================


import hashlib


@dataclass(frozen=True)
class Conflict:
    """Represents a detected conflict with unified model."""

    severity: str  # blocking, warning, info
    conflict_type: str
    entity_id: str
    message: str
    variants: list[dict] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    conflict_id: str = field(default="")
    dedup_key: str = field(default="")
    stage: str = field(default="")
    source_file: str = field(default="")
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.conflict_id:
            id_hash = hashlib.md5(f"{self.conflict_type}-{self.entity_id}-{self.message[:100]}".encode()).hexdigest()
            object.__setattr__(self, 'conflict_id', f"conflict_{id_hash[:16]}")
        
        if not self.dedup_key:
            dedup_hash = hashlib.md5(f"{self.conflict_type}-{self.entity_id}".encode()).hexdigest()
            object.__setattr__(self, 'dedup_key', f"dedup_{dedup_hash[:12]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "dedup_key": self.dedup_key,
            "severity": self.severity,
            "conflict_type": self.conflict_type,
            "entity_id": self.entity_id,
            "stage": self.stage,
            "source_file": self.source_file,
            "message": self.message,
            "variants": self.variants,
            "details": self.details,
            "evidence": self.evidence,
            "detected_at": self.detected_at,
        }



@dataclass
class DetectionContext:
    """Context passed to conflict detection rules."""

    # From fragment (IntegratorJudge style)
    fragment: dict[str, Any] | None = None

    # From indexed data (SemanticOverlapValidator style)
    manifests: list[dict] = field(default_factory=list)
    by_meta_id: dict[str, list[dict]] = field(default_factory=dict)
    by_instance_binding: dict[str, list[dict]] = field(default_factory=dict)
    by_relationship_key: dict[str, list[dict]] = field(default_factory=dict)

    # Configuration
    run_id_prefix: str | None = None


class BaseConflictRule(ABC):
    """Base class for conflict detection rules."""

    name: str = ""
    severity: str = "warning"
    description: str = ""

    @abstractmethod
    def detect(self, context: DetectionContext) -> list[Conflict]:
        """Detect conflicts and return list of Conflict objects."""
        ...


def with_conflict_stage(conflict: Conflict, stage: str) -> Conflict:
    """Return a copy of the conflict with the stage label attached.

    Used by conflict detection pipelines to stamp the originating stage
    onto each conflict before deduplication and reporting. ``Conflict`` is a
    frozen dataclass, so a new instance is returned when the stage changes.
    """
    if not stage or conflict.stage == stage:
        return conflict
    return replace(conflict, stage=stage)


# =============================================================================
# Quality Gate Types (from quality_gate_rules.py)
# =============================================================================


@dataclass
class QualityContext:
    """Context object passed through quality gate rules."""
    fragment: Dict[str, Any]
    domain: str = "SUPPLY_CHAIN"
    iteration: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence(self) -> float:
        """Calculate aggregate confidence from all rule scores."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)
    
    def add_score(self, rule_id: str, score: float) -> None:
        """Add a score from a rule."""
        self.scores[rule_id] = max(0.0, min(1.0, score))
    
    def add_issue(self, issue: str) -> None:
        """Add an issue found by a rule."""
        if issue not in self.issues:
            self.issues.append(issue)


class QualityGateRule(Protocol):
    """Protocol for quality gate rules.
    
    Each rule implements a specific quality check and returns a score [0.0, 1.0].
    Rules are composable and can be chained into pipelines.
    """
    
    rule_id: str
    description: str
    weight: float = 1.0  # Weight in aggregate scoring
    
    async def check(self, context: QualityContext) -> float:
        """Execute the quality check.
        
        Args:
            context: QualityContext containing fragment and accumulated state
            
        Returns:
            Score between 0.0 (fail) and 1.0 (pass)
        """
        ...


class BaseQualityRule(ABC):
    """Base class for quality gate rules."""
    
    rule_id: str = ""
    description: str = ""
    weight: float = 1.0
    
    @abstractmethod
    async def check(self, context: QualityContext) -> float:
        """Execute the quality check. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def _get_nodes(self, fragment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract nodes from fragment.
        
        Supports both dict and Pydantic BaseModel (FragmentNode, etc.).
        Handles multiple formats:
        - Legacy: 'nodes', 'classes'
        - New layered: 'layer1_concepts', 'layer2_scenarios', 'layer25_rules'
        """
        nodes = fragment.get('nodes', []) or fragment.get('classes', []) or []
        
        if not nodes:
            nodes = []
            for key in ['layer1_concepts', 'layer2_scenarios', 'layer25_rules']:
                if key in fragment:
                    nodes.extend(fragment[key])
        
        normalized = []
        for n in nodes:
            if isinstance(n, dict):
                normalized.append(n)
            elif hasattr(n, 'to_dict'):
                normalized.append(n.to_dict())
            elif hasattr(n, 'model_dump'):
                normalized.append(n.model_dump())
            else:
                normalized.append({})
        return normalized
    
    def _get_relationships(self, fragment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationships (handles both 'relationships' and 'edges').
        
        Supports both dict and Pydantic BaseModel (FragmentRelationship, etc.).
        """
        rels = fragment.get('relationships', []) or fragment.get('edges', []) or []
        normalized = []
        for r in rels:
            if isinstance(r, dict):
                normalized.append(r)
            elif hasattr(r, 'to_dict'):
                normalized.append(r.to_dict())
            elif hasattr(r, 'model_dump'):
                normalized.append(r.model_dump())
            else:
                normalized.append({})
        return normalized