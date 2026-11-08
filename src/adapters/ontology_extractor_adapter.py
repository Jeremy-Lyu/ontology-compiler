"""OntologyExtractorPort adapter implementation.

Adapts TrueUnifiedExtractor to OntologyExtractorPort interface.
GEMINI-05: Refactor service layer to adapter implementations.
G26-Cleanup-03: Updated to use TrueUnifiedExtractor (UnifiedExtractor removed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.application.ports.extraction_ports import OntologyExtractorPort, ExtractorResult
from ontology_compiler.services.extraction.true_unified_extractor import TrueUnifiedExtractor


# Compatibility alias for gradual migration
UnifiedExtractor = TrueUnifiedExtractor


class OntologyExtractorAdapter(OntologyExtractorPort):
    """Adapter for TrueUnifiedExtractor implementing OntologyExtractorPort.
    
    This adapter wraps the LLM-based TrueUnifiedExtractor and exposes
    it through the standardized OntologyExtractorPort interface.
    
    Usage:
        adapter = OntologyExtractorAdapter()
        result = await adapter.extract("/path/to/doc.pptx", domain="SUPPLY_CHAIN")
    """
    
    def __init__(self, extractor: TrueUnifiedExtractor | UnifiedExtractor | None = None):
        """Initialize adapter with optional extractor instance.
        
        Args:
            extractor: TrueUnifiedExtractor instance. If None, creates default with LLM.
        """
        if extractor is None:
            # Create default TrueUnifiedExtractor
            self._extractor = TrueUnifiedExtractor()
        else:
            self._extractor = extractor
    
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
            **kwargs: Additional extraction parameters (e.g., refinement_threshold).
            
        Returns:
            ExtractorResult containing nodes, instances, edges.
        """
        path = Path(file_path)
        
        # Read file content
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            content = f"[Error reading file: {e}]"
        
        # Run unified extraction using TrueUnifiedExtractor.extract()
        # TrueUnifiedExtractor.extract() takes content, file_path (optional), existing (optional)
        extraction_result = await self._extractor.extract(
            source=content,
            context={
                "domain": domain,
                "file_path": str(path),
                "use_context": kwargs.get("use_context", True),
                **kwargs
            }
        )
        
        # ExtractorResult from TrueUnifiedExtractor already has the correct format
        # Map to adapter's expected ExtractorResult format if needed
        if hasattr(extraction_result, 'nodes') and hasattr(extraction_result, 'instances'):
            # Already in correct format from TrueUnifiedExtractor
            nodes = extraction_result.nodes
            instances = extraction_result.instances
            edges = extraction_result.edges
            quality_score = extraction_result.metadata.get("quality_score", 0.0) if extraction_result.metadata else 0.0
        else:
            # Legacy format handling (dict-based result)
            l1_data = extraction_result.get("L1", [])
            l2_data = extraction_result.get("L2", [])
            l25_data = extraction_result.get("L2.5", [])
            
            # Combine all nodes (L1 + L2.5 as nodes, L2 as instances)
            nodes = l1_data + [e for e in l25_data if e.get("type") == "technical_entity"]
            instances = l2_data + [e for e in l25_data if e.get("type") != "technical_entity"]
            
            # Extract relationships from all levels
            edges = []
            for entity in l1_data + l2_data + l25_data:
                if "relationships" in entity:
                    edges.extend(entity["relationships"])
            
            # Calculate simple quality score
            quality_score = self._calculate_quality_score(nodes, instances, edges)
        
        return ExtractorResult(
            nodes=nodes,
            instances=instances,
            edges=edges,
            quality_score=quality_score,
            metadata={
                "domain": domain,
                "file_path": file_path,
                "extraction_method": "true_unified_llm",
            }
        )
    
    def supports_file_type(self, file_path: str) -> bool:
        """Check if this extractor supports the given file type.
        
        UnifiedExtractor supports text-based files that can be parsed
        by the document reader (PDF, PPTX, DOCX, MD, etc.).
        
        Args:
            file_path: Path to check.
            
        Returns:
            True if file type is supported.
        """
        supported_extensions = {
            ".pdf", ".pptx", ".docx", ".md", ".txt",
            ".json", ".yaml", ".yml", ".csv", ".xlsx"
        }
        
        path = Path(file_path)
        return path.suffix.lower() in supported_extensions
    
    def _calculate_quality_score(
        self,
        nodes: list[dict],
        instances: list[dict],
        edges: list[dict]
    ) -> float:
        """Calculate a simple quality score based on extraction results.
        
        Higher entity counts and relationship density indicate better extraction.
        
        Args:
            nodes: Extracted nodes.
            instances: Extracted instances.
            edges: Extracted relationships.
            
        Returns:
            Quality score between 0 and 1.
        """
        total_entities = len(nodes) + len(instances)
        
        if total_entities == 0:
            return 0.0
        
        # Score based on entity count (capped at 20 entities for max score)
        entity_score = min(total_entities / 20, 1.0) * 0.6
        
        # Score based on relationship density
        edge_ratio = len(edges) / total_entities if total_entities > 0 else 0
        edge_score = min(edge_ratio / 2, 1.0) * 0.4
        
        return entity_score + edge_score
