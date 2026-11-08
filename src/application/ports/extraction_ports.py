"""Ontology extraction bounded-context port definitions.

Ports and DTOs for ontology extraction, deduplication and conflict
detection. Split from `all_ports.py` during P2 port classification (§9.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractorResult:
    """Result of ontology extraction."""
    nodes: list[dict[str, Any]]
    instances: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    quality_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class OntologyExtractorPort(ABC):
    """Port for ontology extraction from documents.

    Abstracts extraction behavior across rule-based, LLM-based,
    and hybrid extraction strategies.
    """

    @abstractmethod
    async def extract(
        self,
        file_path: str,
        domain: str | None = None,
        **kwargs: Any
    ) -> ExtractorResult:
        """Extract ontology entities from a file.

        Args:
            file_path: Path to the document.
            domain: Optional domain hint.
            **kwargs: Additional extraction parameters.

        Returns:
            ExtractorResult containing nodes, instances, edges.
        """
        raise NotImplementedError

    @abstractmethod
    def supports_file_type(self, file_path: str) -> bool:
        """Check if this extractor supports the given file type."""
        raise NotImplementedError


class DeduplicationPort(ABC):
    """Port for entity deduplication across extraction results."""

    @abstractmethod
    def deduplicate_nodes(
        self,
        nodes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate nodes based on fingerprint/meta_id."""
        raise NotImplementedError

    @abstractmethod
    def deduplicate_edges(
        self,
        edges: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate relationships."""
        raise NotImplementedError

    @abstractmethod
    def compute_fingerprint(self, entity: dict[str, Any]) -> str:
        """Compute unique fingerprint for an entity."""
        raise NotImplementedError


class ConflictDetectionPort(ABC):
    """Port for detecting semantic conflicts in ontology artifacts."""

    @abstractmethod
    def detect_conflicts(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Detect all conflicts (duplicate, circular, drift, semantic).

        Returns:
            List of conflict records with type, severity, and resolution.
        """
        raise NotImplementedError

    @abstractmethod
    def check_circular_references(
        self,
        edges: list[dict[str, Any]]
    ) -> list[list[str]]:
        """Detect circular reference chains."""
        raise NotImplementedError


__all__ = [
    "ExtractorResult",
    "OntologyExtractorPort",
    "DeduplicationPort",
    "ConflictDetectionPort",
]
