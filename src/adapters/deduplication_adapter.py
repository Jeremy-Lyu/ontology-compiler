"""DeduplicationPort adapter implementation.

Adapts EntityDeduplicationService to DeduplicationPort interface.
GEMINI-05: Refactor service layer to adapter implementations.
"""
from __future__ import annotations

from typing import Any

from src.application.ports.extraction_ports import DeduplicationPort, ExtractorResult
from ontology_compiler.services.merge.entity_dedup_service import EntityDeduplicationService


class DeduplicationAdapter(DeduplicationPort):
    """Adapter for EntityDeduplicationService implementing DeduplicationPort.
    
    This adapter wraps the existing EntityDeduplicationService and exposes
    it through the standardized DeduplicationPort interface.
    
    Usage:
        adapter = DeduplicationAdapter()
        unique_nodes = adapter.deduplicate_nodes(nodes)
        unique_edges = adapter.deduplicate_edges(edges)
    """
    
    def __init__(self, service: EntityDeduplicationService | None = None):
        """Initialize adapter with optional service instance.
        
        Args:
            service: EntityDeduplicationService instance. If None, creates default.
        """
        self._service = service or EntityDeduplicationService(strategy="fingerprint")
    
    def deduplicate_nodes(
        self,
        nodes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate nodes based on fingerprint/meta_id.
        
        Args:
            nodes: List of node dictionaries.
            
        Returns:
            List of unique nodes.
        """
        return self._service.deduplicate_nodes(nodes, by="meta_id")
    
    def deduplicate_edges(
        self,
        edges: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate relationships.
        
        Args:
            edges: List of edge dictionaries.
            
        Returns:
            List of unique edges.
        """
        return self._service.deduplicate_edges(edges)
    
    def compute_fingerprint(self, entity: dict[str, Any]) -> str:
        """Compute unique fingerprint for an entity.
        
        Args:
            entity: Entity dictionary.
            
        Returns:
            Fingerprint string.
        """
        return self._service.compute_object_fingerprint(entity)
    
    def deduplicate_result(self, result: ExtractorResult) -> ExtractorResult:
        """Deduplicate all entities in an ExtractorResult.
        
        Convenience method to deduplicate all components of an extraction result.
        
        Args:
            result: ExtractorResult with nodes, instances, edges.
            
        Returns:
            New ExtractorResult with deduplicated entities.
        """
        unique_nodes = self.deduplicate_nodes(result.nodes)
        unique_instances = self.deduplicate_nodes(result.instances)
        unique_edges = self.deduplicate_edges(result.edges)
        
        return ExtractorResult(
            nodes=unique_nodes,
            instances=unique_instances,
            edges=unique_edges,
            quality_score=result.quality_score,
            metadata=result.metadata
        )
