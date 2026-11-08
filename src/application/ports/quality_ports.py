"""Quality & governance bounded-context port definitions.

Ports and DTOs for quality assessment, merge decisions, patch
application, report generation and governance validation. Split from
`all_ports.py` during P2 port classification (§9.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class GovernanceValidatorPort(ABC):
    """Interface for detecting semantic conflicts and drifts in artifacts."""

    @abstractmethod
    def write_report(self, *, runs_root: Path, run_id_prefix: str | None = None) -> dict[str, Any]:
        """Generate and persist a semantic conflict report across multiple runs."""
        raise NotImplementedError


@dataclass
class QualityDimensions:
    """Quality dimensions for ontology assessment."""
    completeness: float = 0.0
    consistency: float = 0.0
    accuracy: float = 0.0
    coverage: float = 0.0
    overall: float = 0.0


class QualityAssessmentPort(ABC):
    """Port for multi-dimensional quality assessment."""

    @abstractmethod
    def calculate_dimensions(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        source_files: list[str],
        **kwargs: Any
    ) -> QualityDimensions:
        """Calculate quality dimensions for extracted entities."""
        raise NotImplementedError

    @abstractmethod
    def should_trigger_refinement(
        self,
        dimensions: QualityDimensions,
        threshold: float = 0.75
    ) -> bool:
        """Determine if quality is below threshold requiring refinement."""
        raise NotImplementedError


@dataclass
class MergeDecision:
    """Decision for entity merging."""
    action: str  # SKIP, MERGE, OVERWRITE, RENAME, FLAG
    target_id: str
    source_id: str
    confidence: float = 0.0
    reason: str = ""


class MergeDecisionPort(ABC):
    """Port for merge decision engine."""

    @abstractmethod
    def decide_merge_action(
        self,
        new_entity: dict[str, Any],
        existing_entities: list[dict[str, Any]]
    ) -> MergeDecision:
        """Decide how to merge new entity with existing master entities."""
        raise NotImplementedError

    @abstractmethod
    def compare_fingerprints(
        self,
        entity1: dict[str, Any],
        entity2: dict[str, Any]
    ) -> float:
        """Compare entity fingerprints, return similarity score [0,1]."""
        raise NotImplementedError


@dataclass
class PatchOperation:
    """Single patch operation."""
    op: str  # ADD, DELETE, MODIFY
    entity_type: str  # node, edge, instance
    entity_id: str
    payload: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None


class PatchApplicationPort(ABC):
    """Port for applying ontology patches."""

    def set_paths(self, l1_path: Path, l2_path: Path, manifest_path: Path) -> None:
        """Configure file paths used during patch application.

        Implementations that do not need file paths may ignore this hook.
        """
        return None

    @abstractmethod
    def apply_patch(
        self,
        operations: list[PatchOperation],
        dry_run: bool = False
    ) -> dict[str, Any]:
        """Apply patch operations to ontology.

        Args:
            operations: List of patch operations.
            dry_run: If True, validate without applying.

        Returns:
            Result with success status and applied changes.
        """
        raise NotImplementedError

    @abstractmethod
    def rollback(self, patch_id: str) -> bool:
        """Rollback a previously applied patch."""
        raise NotImplementedError


class ReportGenerationPort(ABC):
    """Port for generating ontology analysis reports."""

    @abstractmethod
    def generate_quality_briefing(
        self,
        run_id: str,
        dimensions: QualityDimensions,
        **kwargs: Any
    ) -> str:
        """Generate quality briefing report (Markdown)."""
        raise NotImplementedError

    @abstractmethod
    def render_core_results(
        self,
        l1_data: dict[str, Any],
        l2_data: dict[str, Any],
        **kwargs: Any
    ) -> str:
        """Render core results as Markdown."""
        raise NotImplementedError

    @abstractmethod
    def generate_business_scenarios(
        self,
        l1_data: dict[str, Any],
        l2_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate L2 business scenarios from ontology data."""
        raise NotImplementedError


__all__ = [
    "GovernanceValidatorPort",
    "QualityDimensions",
    "QualityAssessmentPort",
    "MergeDecision",
    "MergeDecisionPort",
    "PatchOperation",
    "PatchApplicationPort",
    "ReportGenerationPort",
]
