"""Quality assurance service boundary.

This package is a grouping boundary for quality reviewers, gates, rules, and
relation governance. Canonical modules remain in the parent package for
compatibility during migration.
"""
from __future__ import annotations

from ..relation_governance import *
from ..skill_telemetry import *

from .base import BaseQualityRule, Conflict, QualityContext, with_conflict_stage
from .conflict_dedup import (
    ConflictDedupService,
    ConflictStats,
    dedup_conflicts,
    merge_conflicts,
    calculate_stats,
)
from .pipelines import QualityGatePipeline, PipelineResult
from .quality_rules import (
    NodeCompletenessRule,
    RelationshipValidityRule,
    SemanticQualityRule,
    DomainAlignmentRule,
    CrossLayerConsistencyRule,
    DuplicateDetectionRule,
    LayerBalanceRule,
    FieldCompletenessRule,
    LayerInstanceDisciplineRule,
    NamingConsistencyRule,
    EvidenceCoverageRule,
)

__all__ = [
    # Governance & telemetry
    "RelationGovernanceEngine",
    "get_relation_governance_engine",
    # Conflict base & dedup
    "Conflict",
    "with_conflict_stage",
    "ConflictDedupService",
    "ConflictStats",
    "dedup_conflicts",
    "merge_conflicts",
    "calculate_stats",
    # Rule abstractions
    "BaseQualityRule",
    "QualityContext",
    # Pipeline & result
    "QualityGatePipeline",
    "PipelineResult",
    # Built-in rules
    "NodeCompletenessRule",
    "RelationshipValidityRule",
    "SemanticQualityRule",
    "DomainAlignmentRule",
    "CrossLayerConsistencyRule",
    "DuplicateDetectionRule",
    "LayerBalanceRule",
    "FieldCompletenessRule",
    "LayerInstanceDisciplineRule",
    "NamingConsistencyRule",
    "EvidenceCoverageRule",
]
