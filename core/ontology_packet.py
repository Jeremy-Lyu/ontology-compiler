"""Ontology Packet - Unified output format for ontology extraction pipeline.

Based on schemas/ontology_packet.schema.json, this module provides:
1. OntologyEntity - Standardized entity model
2. OntologyRelation - Standardized relation model  
3. OntologyEvidence - Standardized evidence model
4. OntologyPacket - Container for extraction results
5. DocumentInfo - Source document metadata

This format ensures:
- Consistent entity/relation structure across all extraction layers
- Provenance tracking for traceability
- Evidence binding for quality assurance
- RAG-friendly output structure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class DocumentInfo:
    """Source document information."""
    file_name: str
    file_path: str
    domain: str
    document_type: Optional[str] = None
    language: str = "zh"
    size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "file_name": self.file_name,
            "file_path": self.file_path,
            "domain": self.domain,
            "language": self.language,
        }
        if self.document_type:
            result["document_type"] = self.document_type
        if self.size_bytes:
            result["size_bytes"] = self.size_bytes
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            result["updated_at"] = self.updated_at.isoformat()
        return result


@dataclass
class OntologyEntity:
    """Standardized ontology entity."""
    id: str
    name: str
    layer: str
    type: str
    
    code: Optional[str] = None
    sub_type: Optional[str] = None
    status: str = "active"
    parent_id: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    source_file: Optional[str] = None
    source_span: Optional[str] = None
    evidence_ref: Optional[str] = None
    version: str = "1.0"
    confidence: float = 0.8
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    relations_summary: Dict[str, int] = field(default_factory=dict)
    layer_scope: str = "spec"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "layer": self.layer,
            "type": self.type,
            "status": self.status,
            "version": self.version,
            "confidence": self.confidence,
            "layer_scope": self.layer_scope,
            "attributes": self.attributes,
            "aliases": self.aliases,
            "tags": self.tags,
            "relations_summary": self.relations_summary,
        }
        if self.code:
            result["code"] = self.code
        if self.sub_type:
            result["sub_type"] = self.sub_type
        if self.parent_id:
            result["parent_id"] = self.parent_id
        if self.description:
            result["description"] = self.description
        if self.domain:
            result["domain"] = self.domain
        if self.source_file:
            result["source_file"] = self.source_file
        if self.source_span:
            result["source_span"] = self.source_span
        if self.evidence_ref:
            result["evidence_ref"] = self.evidence_ref
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            result["updated_at"] = self.updated_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyEntity":
        return cls(
            id=data["id"],
            name=data["name"],
            layer=data["layer"],
            type=data["type"],
            code=data.get("code"),
            sub_type=data.get("sub_type"),
            status=data.get("status", "active"),
            parent_id=data.get("parent_id"),
            description=data.get("description"),
            domain=data.get("domain"),
            source_file=data.get("source_file"),
            source_span=data.get("source_span"),
            evidence_ref=data.get("evidence_ref"),
            version=data.get("version", "1.0"),
            confidence=data.get("confidence", 0.8),
            attributes=data.get("attributes", {}),
            aliases=data.get("aliases", []),
            tags=data.get("tags", []),
            relations_summary=data.get("relations_summary", {}),
            layer_scope=data.get("layer_scope", "spec"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )


@dataclass
class OntologyRelation:
    """Standardized ontology relation."""
    source_id: str
    target_id: str
    relation_type: str
    
    layer_scope: str = "spec"
    confidence: float = 0.8
    evidence_ref: Optional[str] = None
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None
    provenance: str = "llm"
    description: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "layer_scope": self.layer_scope,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "attributes": self.attributes,
        }
        if self.evidence_ref:
            result["evidence_ref"] = self.evidence_ref
        if self.effective_from:
            result["effective_from"] = self.effective_from.isoformat()
        if self.effective_to:
            result["effective_to"] = self.effective_to.isoformat()
        if self.description:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyRelation":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            layer_scope=data.get("layer_scope", "spec"),
            confidence=data.get("confidence", 0.8),
            evidence_ref=data.get("evidence_ref"),
            effective_from=datetime.fromisoformat(data["effective_from"]) if data.get("effective_from") else None,
            effective_to=datetime.fromisoformat(data["effective_to"]) if data.get("effective_to") else None,
            provenance=data.get("provenance", "llm"),
            description=data.get("description"),
            attributes=data.get("attributes", {}),
        )


@dataclass
class EvidencePosition:
    """Position information for evidence."""
    page: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.page:
            result["page"] = self.page
        if self.start_line:
            result["start_line"] = self.start_line
        if self.end_line:
            result["end_line"] = self.end_line
        if self.start_char:
            result["start_char"] = self.start_char
        if self.end_char:
            result["end_char"] = self.end_char
        return result


@dataclass
class OntologyEvidence:
    """Standardized evidence model."""
    id: str
    text: str
    document: str
    
    position: Optional[EvidencePosition] = None
    version: Optional[str] = None
    entity_ids: List[str] = field(default_factory=list)
    relation_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "text": self.text,
            "document": self.document,
            "entity_ids": self.entity_ids,
            "relation_ids": self.relation_ids,
        }
        if self.position:
            result["position"] = self.position.to_dict()
        if self.version:
            result["version"] = self.version
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyEvidence":
        position_data = data.get("position")
        position = EvidencePosition(
            page=position_data.get("page") if position_data else None,
            start_line=position_data.get("start_line") if position_data else None,
            end_line=position_data.get("end_line") if position_data else None,
            start_char=position_data.get("start_char") if position_data else None,
            end_char=position_data.get("end_char") if position_data else None,
        ) if position_data else None
        
        return cls(
            id=data["id"],
            text=data["text"],
            document=data["document"],
            position=position,
            version=data.get("version"),
            entity_ids=data.get("entity_ids", []),
            relation_ids=data.get("relation_ids", []),
        )


@dataclass
class ProvenanceInfo:
    """Provenance information for traceability."""
    extraction_time: datetime
    extractor_version: Optional[str] = None
    llm_model: Optional[str] = None
    prompt_version: Optional[str] = None
    skills_used: List[str] = field(default_factory=list)
    agent_name: Optional[str] = None
    confidence_threshold: Optional[float] = None
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "extraction_time": self.extraction_time.isoformat(),
            "skills_used": self.skills_used,
            "retries": self.retries,
        }
        if self.extractor_version:
            result["extractor_version"] = self.extractor_version
        if self.llm_model:
            result["llm_model"] = self.llm_model
        if self.prompt_version:
            result["prompt_version"] = self.prompt_version
        if self.agent_name:
            result["agent_name"] = self.agent_name
        if self.confidence_threshold:
            result["confidence_threshold"] = self.confidence_threshold
        return result


@dataclass
class QualityMetrics:
    """Quality metrics for the extraction."""
    entity_completeness: Optional[float] = None
    relation_validity: Optional[float] = None
    evidence_coverage: Optional[float] = None
    duplicate_rate: Optional[float] = None
    conflict_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        result = {"conflict_count": self.conflict_count}
        if self.entity_completeness is not None:
            result["entity_completeness"] = self.entity_completeness
        if self.relation_validity is not None:
            result["relation_validity"] = self.relation_validity
        if self.evidence_coverage is not None:
            result["evidence_coverage"] = self.evidence_coverage
        if self.duplicate_rate is not None:
            result["duplicate_rate"] = self.duplicate_rate
        return result


@dataclass
class ExtractionWarning:
    """Warning message from extraction process."""
    code: str
    message: str
    severity: str = "warning"
    entity_id: Optional[str] = None
    relation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.entity_id:
            result["entity_id"] = self.entity_id
        if self.relation_id:
            result["relation_id"] = self.relation_id
        return result


@dataclass
class OntologyPacket:
    """Unified container for ontology extraction results."""
    document: DocumentInfo
    entities: Dict[str, List[OntologyEntity]] = field(default_factory=lambda: {
        "L0": [], "L1": [], "L2": [], "L2.5": [], "L3": []
    })
    relations: List[OntologyRelation] = field(default_factory=list)
    evidence: List[OntologyEvidence] = field(default_factory=list)
    layer_scope: str = "spec"
    provenance: Optional[ProvenanceInfo] = None
    quality: Optional[QualityMetrics] = None
    warnings: List[ExtractionWarning] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    def add_entity(self, entity: OntologyEntity) -> None:
        """Add an entity to the appropriate layer."""
        if entity.layer not in self.entities:
            self.entities[entity.layer] = []
        self.entities[entity.layer].append(entity)

    def add_relation(self, relation: OntologyRelation) -> None:
        """Add a relation."""
        self.relations.append(relation)

    def add_evidence(self, evidence: OntologyEvidence) -> None:
        """Add evidence."""
        self.evidence.append(evidence)

    def add_warning(self, warning: ExtractionWarning) -> None:
        """Add a warning."""
        self.warnings.append(warning)

    def get_entity_count(self) -> int:
        """Get total entity count across all layers."""
        return sum(len(layer_entities) for layer_entities in self.entities.values())

    def get_entity_by_id(self, entity_id: str) -> Optional[OntologyEntity]:
        """Find entity by ID across all layers."""
        for layer_entities in self.entities.values():
            for entity in layer_entities:
                if entity.id == entity_id:
                    return entity
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "document": self.document.to_dict(),
            "layer_scope": self.layer_scope,
            "entities": {
                layer: [entity.to_dict() for entity in entities]
                for layer, entities in self.entities.items()
            },
            "relations": [relation.to_dict() for relation in self.relations],
            "evidence": [evidence.to_dict() for evidence in self.evidence],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "assumptions": self.assumptions,
        }
        if self.provenance:
            result["provenance"] = self.provenance.to_dict()
        if self.quality:
            result["quality"] = self.quality.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyPacket":
        """Create from dictionary."""
        document = DocumentInfo(**data["document"])
        
        entities = {}
        for layer, layer_data in data.get("entities", {}).items():
            entities[layer] = [OntologyEntity.from_dict(e) for e in layer_data]
        
        relations = [OntologyRelation.from_dict(r) for r in data.get("relations", [])]
        evidence = [OntologyEvidence.from_dict(e) for e in data.get("evidence", [])]
        
        provenance_data = data.get("provenance")
        provenance = ProvenanceInfo(
            extraction_time=datetime.fromisoformat(provenance_data["extraction_time"]),
            extractor_version=provenance_data.get("extractor_version"),
            llm_model=provenance_data.get("llm_model"),
            prompt_version=provenance_data.get("prompt_version"),
            skills_used=provenance_data.get("skills_used", []),
            agent_name=provenance_data.get("agent_name"),
            confidence_threshold=provenance_data.get("confidence_threshold"),
            retries=provenance_data.get("retries", 0),
        ) if provenance_data else None
        
        quality_data = data.get("quality")
        quality = QualityMetrics(**quality_data) if quality_data else None
        
        warnings = [ExtractionWarning(**w) for w in data.get("warnings", [])]
        
        return cls(
            document=document,
            entities=entities,
            relations=relations,
            evidence=evidence,
            layer_scope=data.get("layer_scope", "spec"),
            provenance=provenance,
            quality=quality,
            warnings=warnings,
            assumptions=data.get("assumptions", []),
        )