"""Layered Artifact Standardization for Entity Extraction.

[DIAG-3.1] Standardize layered artifacts for incremental updates and traceability.

[BATCH 4 CONSOLIDATION 2026-06-29]:
- This module provides **layered data models and storage** for L1/L2/L2.5 artifacts
- For **generic artifact I/O operations**, see artifacts.py

[UNIFIED MODEL 2026-07-10]:
- NormalizedEntity now inherits from CanonicalEntity
- RelationshipArtifact now inherits from CanonicalRelation
- Reduces dict conversion overhead across pipeline

Boundary Alignment:
- layer_artifacts.py: Data models (LLMRawResponse, NormalizedEntity, etc.), LayerArtifactStore
- artifacts.py: Generic JSON/text writing, schema validation, auxiliary relocation
- core/models/canonical.py: Unified CanonicalEntity/CanonicalRelation base models

This module is re-exported from artifacts.py for unified access.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.domain.models import Evidence, CanonicalEntity, CanonicalRelation, CanonicalAttribute
from pydantic import BaseModel, Field

from core.contracts import CacheVersionBundle
from .artifacts import write_json_artifact

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """Types of artifacts."""
    RAW = "raw"
    NORMALIZED = "normalized"
    RELATIONSHIP = "relationship"
    QUALITY = "quality"
    CONFIDENCE = "confidence"


class RelationScope(Enum):
    """Scope of relationships."""
    INTRA_FILE = "intra_file"
    INTER_FILE = "inter_file"
    FOLDER_LEVEL = "folder_level"


@dataclass
class LLMRawResponse:
    """Raw LLM response for traceability."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    temperature: float
    response_text: str
    timestamp: datetime
    strategy: str  # e.g., "technical", "business"
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "temperature": self.temperature,
            "response_text": self.response_text,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


class NormalizedEntity(CanonicalEntity):
    """Normalized entity after deduplication and canonicalization.

    [UNIFIED MODEL]: Inherits from CanonicalEntity to reduce dict conversion.

    Evidence Integration (2026-07-06):
    - evidence field now stores full Evidence objects for traceability
    - source_files remains for backward compatibility

    Additional fields beyond CanonicalEntity:
    - meta_subclass: Optional subclass classification
    - canonical_id: ID after deduplication merging
    - quality_score: Combined quality assessment score
    - version: For incremental updates tracking
    """
    meta_subclass: Optional[str] = None
    canonical_id: Optional[str] = None
    quality_score: float = 0.0
    version: int = 1

    @classmethod
    def from_canonical(cls, entity: CanonicalEntity) -> "NormalizedEntity":
        """Create NormalizedEntity from CanonicalEntity."""
        return cls(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            layer=entity.layer,
            status=entity.status,
            description=entity.description,
            aliases=entity.aliases,
            properties=entity.properties,
            attributes=entity.attributes,
            evidence=entity.evidence,
            source_trace=entity.source_trace,
            confidence=entity.confidence,
            version=str(entity.lineage.extraction_run_id) if entity.lineage.extraction_run_id else "1",
            lineage=entity.lineage,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            metadata=entity.metadata,
            meta_subclass=None,
            canonical_id=entity.entity_id,
            quality_score=entity.confidence,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizedEntity":
        """Create NormalizedEntity from dictionary (backward compatibility)."""
        canonical = CanonicalEntity.from_dict(data)
        return cls(
            entity_id=canonical.entity_id,
            name=canonical.name,
            entity_type=canonical.entity_type,
            layer=canonical.layer,
            status=canonical.status,
            description=canonical.description,
            aliases=canonical.aliases,
            properties=canonical.properties,
            attributes=canonical.attributes,
            evidence=canonical.evidence,
            source_trace=canonical.source_trace,
            confidence=canonical.confidence,
            version=canonical.version,
            lineage=canonical.lineage,
            created_at=canonical.created_at,
            updated_at=canonical.updated_at,
            metadata=canonical.metadata,
            meta_subclass=data.get("meta_subclass"),
            canonical_id=data.get("canonical_id", data.get("entity_id", data.get("meta_id"))),
            quality_score=data.get("quality_score", canonical.confidence),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with additional fields."""
        result = super().model_dump()
        result.update({
            "meta_id": self.entity_id,
            "meta_name": self.name,
            "meta_class": self.entity_type,
            "meta_subclass": self.meta_subclass,
            "canonical_id": self.canonical_id,
            "quality_score": round(self.quality_score, 3),
            "version": self.version,
        })
        return result


class RelationshipArtifact(CanonicalRelation):
    """Relationship between entities with scope tracking.

    [UNIFIED MODEL]: Inherits from CanonicalRelation to reduce dict conversion.

    Evidence Integration (2026-07-06):
    - evidence field now stores full Evidence objects for traceability
    - source_file remains for backward compatibility

    Additional fields beyond CanonicalRelation:
    - scope: INTRA_FILE, INTER_FILE, FOLDER_LEVEL
    - layer_span: e.g., "L1_L1", "L1_L2", "L2_L2.5"
    - source_file: Source file path for backward compatibility
    """
    from_id: str = ""
    to_id: str = ""
    scope: RelationScope = RelationScope.INTRA_FILE
    layer_span: str = ""
    source_file: Optional[str] = None

    @classmethod
    def from_canonical(cls, relation: CanonicalRelation) -> "RelationshipArtifact":
        """Create RelationshipArtifact from CanonicalRelation."""
        return cls(
            relation_id=relation.relation_id,
            source_id=relation.source_id,
            target_id=relation.target_id,
            relation_type=relation.relation_type,
            description=relation.description,
            properties=relation.properties,
            evidence=relation.evidence,
            confidence=relation.confidence,
            is_directed=relation.is_directed,
            weight=relation.weight,
            layer=relation.layer,
            status=relation.status,
            created_at=relation.created_at,
            metadata=relation.metadata,
            from_id=relation.source_id,
            to_id=relation.target_id,
            scope=RelationScope.INTRA_FILE,
            layer_span=f"{relation.layer}_{relation.layer}",
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipArtifact":
        """Create RelationshipArtifact from dictionary (backward compatibility)."""
        canonical = CanonicalRelation.from_dict(data)
        scope_value = data.get("scope", "intra_file")
        if isinstance(scope_value, str):
            try:
                scope = RelationScope(scope_value)
            except ValueError:
                scope = RelationScope.INTRA_FILE
        else:
            scope = RelationScope.INTRA_FILE

        return cls(
            relation_id=canonical.relation_id,
            source_id=canonical.source_id,
            target_id=canonical.target_id,
            relation_type=canonical.relation_type,
            description=canonical.description,
            properties=canonical.properties,
            evidence=canonical.evidence,
            confidence=canonical.confidence,
            is_directed=canonical.is_directed,
            weight=canonical.weight,
            layer=canonical.layer,
            status=canonical.status,
            created_at=canonical.created_at,
            metadata=canonical.metadata,
            from_id=data.get("from_id", canonical.source_id),
            to_id=data.get("to_id", canonical.target_id),
            scope=scope,
            layer_span=data.get("layer_span", f"{canonical.layer}_{canonical.layer}"),
            source_file=data.get("source_file"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with additional fields."""
        result = super().model_dump()
        result.update({
            "from_id": self.from_id or self.source_id,
            "to_id": self.to_id or self.target_id,
            "scope": self.scope.value,
            "layer_span": self.layer_span,
            "source_file": self.source_file,
            "weight": round(self.weight, 3),
        })
        return result


@dataclass
class QualityMetrics:
    """Quality metrics for extraction results."""
    entity_count: int = 0
    relation_count: int = 0
    avg_confidence: float = 0.0
    avg_quality: float = 0.0
    low_confidence_count: int = 0
    low_quality_count: int = 0
    duplicate_count: int = 0
    conflict_count: int = 0
    coverage_ratio: float = 0.0  # Percentage of expected entities found
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_quality": round(self.avg_quality, 3),
            "low_confidence_count": self.low_confidence_count,
            "low_quality_count": self.low_quality_count,
            "duplicate_count": self.duplicate_count,
            "conflict_count": self.conflict_count,
            "coverage_ratio": round(self.coverage_ratio, 3),
        }


class LayerArtifact(BaseModel):
    """Complete artifact set for a single layer.

    [UNIFIED MODEL]: Now uses Pydantic BaseModel for better serialization.

    P1 / C-4: ``version_bundle`` records the exact 5-dimension version key
    (prompt / schema / policy / model_family / extraction_mode) that produced
    this artifact. Cross-run incremental reuse can compare bundles to decide
    *which* dimensions changed and whether the artifact is still valid.
    """
    layer: str = "L1"
    raw_response: Optional[LLMRawResponse] = None
    normalized_entities: List[NormalizedEntity] = Field(default_factory=list)
    relationships: List[RelationshipArtifact] = Field(default_factory=list)
    quality: Optional[QualityMetrics] = None
    version: int = 1
    timestamp: datetime = Field(default_factory=datetime.now)
    # P1 / C-4: structured version bundle for precise cross-run reuse decisions
    version_bundle: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "raw_response": self.raw_response.to_dict() if self.raw_response else None,
            "normalized_entities": [e.to_dict() for e in self.normalized_entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "quality": self.quality.to_dict() if self.quality else None,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "version_bundle": self.version_bundle,
        }


class FolderLevelArtifact(BaseModel):
    """Folder-level artifacts for cross-file relationships and global metrics.

    [UNIFIED MODEL]: Now uses Pydantic BaseModel for better serialization.

    P1 / C-4: ``version_bundle`` records the 5-dimension version key for the
    folder-level processing, enabling precise cross-run reuse decisions.
    """
    folder_path: str = ""
    run_id: str = ""
    layer_artifacts: Dict[str, LayerArtifact] = Field(default_factory=dict)
    folder_level_relationships: List[RelationshipArtifact] = Field(default_factory=list)
    global_quality: Optional[QualityMetrics] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    # P1 / C-4: structured version bundle for cross-run reuse decisions
    version_bundle: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folder_path": self.folder_path,
            "run_id": self.run_id,
            "layer_artifacts": {
                layer: artifact.to_dict()
                for layer, artifact in self.layer_artifacts.items()
            },
            "folder_level_relationships": [r.to_dict() for r in self.folder_level_relationships],
            "global_quality": self.global_quality.to_dict() if self.global_quality else None,
            "timestamp": self.timestamp.isoformat(),
            "version_bundle": self.version_bundle,
        }


class ArtifactStage(Enum):
    """Stage of artifact processing."""
    RAW = "raw"
    GOVERNED = "governed"


class LayerArtifactStore:
    """
    [DIAG-3.1] Store and retrieve layered artifacts.
    
    Features:
    - Separate files for each artifact type
    - Support for incremental updates
    - Version tracking for each entity
    - Efficient loading of specific artifacts
    - Raw/governed layered storage for data continuity
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize artifact store.
        
        Args:
            base_dir: Base directory for artifact storage
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = base_dir / "raw"
        self.governed_dir = base_dir / "governed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.governed_dir.mkdir(parents=True, exist_ok=True)
    
    def store_layer_artifact(
        self,
        run_id: str,
        layer: str,
        artifact: LayerArtifact,
        stage: ArtifactStage = ArtifactStage.GOVERNED,
    ) -> Path:
        """
        Store a layer artifact.
        
        Args:
            run_id: Unique run identifier
            layer: Layer name (L1, L2, L2.5)
            artifact: LayerArtifact to store
            stage: Artifact stage (RAW or GOVERNED)
            
        Returns:
            Path to stored artifact file
        """
        base_dir = self.raw_dir if stage == ArtifactStage.RAW else self.governed_dir
        artifact_dir = base_dir / run_id / "layers" / layer
        
        # Store raw response
        if artifact.raw_response:
            write_json_artifact(run_dir=artifact_dir, filename="raw_response.json", payload=artifact.raw_response.to_dict())
        
        # Store normalized entities
        write_json_artifact(
            run_dir=artifact_dir,
            filename="normalized_entities.json",
            payload=[e.to_dict() for e in artifact.normalized_entities],
        )
        
        # Store relationships
        write_json_artifact(
            run_dir=artifact_dir,
            filename="relationships.json",
            payload=[r.to_dict() for r in artifact.relationships],
        )
        
        # Store quality metrics
        if artifact.quality:
            write_json_artifact(run_dir=artifact_dir, filename="quality.json", payload=artifact.quality.to_dict())
        
        # Store manifest
        manifest_path = write_json_artifact(run_dir=artifact_dir, filename="manifest.json", payload=artifact.to_dict())
        
        logger.info(f"Stored layer artifact ({stage.value}) for {layer} at {artifact_dir}")
        return Path(manifest_path)
    
    def load_layer_artifact(
        self,
        run_id: str,
        layer: str,
        artifact_type: Optional[ArtifactType] = None,
        stage: ArtifactStage = ArtifactStage.GOVERNED,
    ) -> Optional[Any]:
        """
        Load a layer artifact or specific artifact type.
        
        Args:
            run_id: Unique run identifier
            layer: Layer name (L1, L2, L2.5)
            artifact_type: Specific artifact type to load (optional)
            stage: Artifact stage (RAW or GOVERNED)
            
        Returns:
            Loaded artifact data, or None if not found
        """
        base_dir = self.raw_dir if stage == ArtifactStage.RAW else self.governed_dir
        artifact_dir = base_dir / run_id / "layers" / layer
        
        if not artifact_dir.exists():
            logger.warning(f"Artifact directory not found: {artifact_dir}")
            return None
        
        if artifact_type is None:
            # Load complete artifact
            manifest_file = artifact_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        
        # Load specific artifact type
        if artifact_type == ArtifactType.RAW:
            file_path = artifact_dir / "raw_response.json"
        elif artifact_type == ArtifactType.NORMALIZED:
            file_path = artifact_dir / "normalized_entities.json"
        elif artifact_type == ArtifactType.RELATIONSHIP:
            file_path = artifact_dir / "relationships.json"
        elif artifact_type == ArtifactType.QUALITY:
            file_path = artifact_dir / "quality.json"
        else:
            return None
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def store_folder_artifact(
        self,
        artifact: FolderLevelArtifact,
        stage: ArtifactStage = ArtifactStage.GOVERNED,
    ) -> Path:
        """
        Store folder-level artifact.
        
        Args:
            artifact: FolderLevelArtifact to store
            stage: Artifact stage (RAW or GOVERNED)
            
        Returns:
            Path to stored artifact file
        """
        base_dir = self.raw_dir if stage == ArtifactStage.RAW else self.governed_dir
        run_dir = base_dir / artifact.run_id
        
        # Store folder-level relationships
        write_json_artifact(
            run_dir=run_dir,
            filename="folder_relationships.json",
            payload=[r.to_dict() for r in artifact.folder_level_relationships],
        )
        
        # Store global quality
        if artifact.global_quality:
            write_json_artifact(run_dir=run_dir, filename="global_quality.json", payload=artifact.global_quality.to_dict())
        
        # Store complete manifest
        manifest_path = write_json_artifact(run_dir=run_dir, filename="folder_manifest.json", payload=artifact.to_dict())
        
        logger.info(f"Stored folder artifact ({stage.value}) for {artifact.folder_path} at {run_dir}")
        return Path(manifest_path)
    
    def load_folder_artifact(
        self,
        run_id: str,
        artifact_type: Optional[ArtifactType] = None,
        stage: ArtifactStage = ArtifactStage.GOVERNED,
    ) -> Optional[Any]:
        """
        Load folder-level artifact or specific artifact type.
        
        Args:
            run_id: Unique run identifier
            artifact_type: Specific artifact type to load (optional)
            stage: Artifact stage (RAW or GOVERNED)
            
        Returns:
            Loaded artifact data, or None if not found
        """
        base_dir = self.raw_dir if stage == ArtifactStage.RAW else self.governed_dir
        run_dir = base_dir / run_id
        
        if not run_dir.exists():
            logger.warning(f"Run directory not found: {run_dir}")
            return None
        
        if artifact_type is None:
            # Load complete artifact
            manifest_file = run_dir / "folder_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        
        # Load specific artifact type
        if artifact_type == ArtifactType.RELATIONSHIP:
            file_path = run_dir / "folder_relationships.json"
        elif artifact_type == ArtifactType.QUALITY:
            file_path = run_dir / "global_quality.json"
        else:
            return None
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_artifact_versions(self, run_id: str, layer: str) -> Dict[str, int]:
        """
        Get version information for artifacts in a layer.
        
        Args:
            run_id: Unique run identifier
            layer: Layer name (L1, L2, L2.5)
            
        Returns:
            Dictionary of entity IDs to their versions
        """
        entities = self.load_layer_artifact(run_id, layer, ArtifactType.NORMALIZED)
        if not entities:
            return {}
        
        versions = {}
        for entity in entities:
            versions[entity["meta_id"]] = entity.get("version", 1)
        return versions
    
    def incremental_update(
        self,
        run_id: str,
        layer: str,
        updated_entities: List[NormalizedEntity],
    ) -> Path:
        """
        Perform incremental update for a layer.
        
        This updates only the specified entities while preserving others.
        
        Args:
            run_id: Unique run identifier
            layer: Layer name (L1, L2, L2.5)
            updated_entities: List of entities to update
            
        Returns:
            Path to updated artifact file
        """
        # Load existing entities
        existing_data = self.load_layer_artifact(run_id, layer)
        if existing_data:
            # Handle both dict (from JSON) and LayerArtifact objects
            if isinstance(existing_data, dict):
                existing_entities = [
                    NormalizedEntity.from_dict(e)
                    for e in existing_data.get("normalized_entities", [])
                ]
            else:
                existing_entities = existing_data.normalized_entities
            
            entity_map = {e.entity_id: e for e in existing_entities}
        else:
            entity_map = {}
        
        # Update entities with new versions
        for entity in updated_entities:
            if entity.entity_id in entity_map:
                entity.version = entity_map[entity.entity_id].version + 1
            entity_map[entity.entity_id] = entity
        
        # Create updated artifact
        updated_artifact = LayerArtifact(
            layer=layer,
            normalized_entities=list(entity_map.values()),
        )
        
        # Store updated artifact
        return self.store_layer_artifact(run_id, layer, updated_artifact)
    
    def list_runs(self, stage: ArtifactStage = ArtifactStage.GOVERNED) -> List[str]:
        """List all run IDs in the store."""
        base_dir = self.raw_dir if stage == ArtifactStage.RAW else self.governed_dir
        return [d.name for d in base_dir.iterdir() if d.is_dir()]

    def promote_to_governed(
        self,
        run_id: str,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Promote artifacts from RAW to GOVERNED stage.
        
        This moves verified artifacts from raw storage to governed storage.
        
        Args:
            run_id: Unique run identifier
            layers: List of layers to promote (optional, defaults to all)
            
        Returns:
            Dictionary of layer names to promoted artifact paths
        """
        promoted_paths = {}
        
        if layers is None:
            layers = ["L0", "L1", "L2", "L2.5", "L3"]
        
        for layer in layers:
            raw_artifact_data = self.load_layer_artifact(
                run_id, layer, stage=ArtifactStage.RAW
            )
            if raw_artifact_data:
                artifact = LayerArtifact(
                    layer=raw_artifact_data.get("layer", layer),
                    normalized_entities=[
                        NormalizedEntity.from_dict(e)
                        for e in raw_artifact_data.get("normalized_entities", [])
                    ],
                    relationships=[
                        RelationshipArtifact.from_dict(r)
                        for r in raw_artifact_data.get("relationships", [])
                    ],
                    quality=QualityMetrics(
                        **raw_artifact_data.get("quality", {})
                    ) if raw_artifact_data.get("quality") else None,
                    version=raw_artifact_data.get("version", 1),
                )
                
                path = self.store_layer_artifact(run_id, layer, artifact, stage=ArtifactStage.GOVERNED)
                promoted_paths[layer] = path
                logger.info(f"Promoted {layer} artifact to GOVERNED stage")
        
        return promoted_paths


def demo_layer_artifacts():
    """Demonstrate layer artifact usage."""
    import tempfile
    from pathlib import Path
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        store = LayerArtifactStore(base_dir)
        run_id = "demo_run_001"
        
        # Create sample L1 artifact
        l1_entity = NormalizedEntity(
            meta_id="L1-001",
            meta_name="Supply Chain",
            meta_class="Concept",
            layer="L1",
            description="The flow of goods and services",
            quality_score=0.85,
            confidence=0.92,
        )
        
        l1_artifact = LayerArtifact(
            layer="L1",
            normalized_entities=[l1_entity],
            quality=QualityMetrics(entity_count=1, avg_quality=0.85),
        )
        
        # Store artifact
        manifest_path = store.store_layer_artifact(run_id, "L1", l1_artifact)
        print(f"Stored L1 artifact at: {manifest_path}")
        
        # Load artifact
        loaded = store.load_layer_artifact(run_id, "L1")
        print(f"Loaded L1 artifact: {len(loaded['normalized_entities'])} entities")
        
        # Incremental update
        updated_entity = NormalizedEntity(
            meta_id="L1-001",
            meta_name="Supply Chain (Updated)",
            meta_class="Concept",
            layer="L1",
            description="The flow of goods and services (updated)",
            quality_score=0.90,
            confidence=0.95,
        )
        
        updated_path = store.incremental_update(run_id, "L1", [updated_entity])
        print(f"Updated L1 artifact at: {updated_path}")
        
        # Check versions
        versions = store.get_artifact_versions(run_id, "L1")
        print(f"Entity versions: {versions}")


if __name__ == "__main__":
    demo_layer_artifacts()