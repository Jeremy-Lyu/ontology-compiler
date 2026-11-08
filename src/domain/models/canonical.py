"""Canonical Models - Unified intermediate object models.

This module is the canonical source for Canonical models.
Shared between src and ontology_compiler to avoid duplication.

[Phase 1 - P0-UNI 2026-06-18]: Refactored to Pydantic, unified type system

This module defines the unified output structure for all extraction modules,
ensuring:
1. Consistent output format across all extractors
2. Direct consumption by downstream consumers (RAG, reporting, QA)
3. Complete unification of entity/relation/attribute/evidence intermediate representations
4. Standardized version management and lineage tracking

Architecture goals:
- G1: Unified knowledge extraction protocol
- G4: Unified knowledge consumption layer
- G5: Unified quality闭环

Example:
    result = CanonicalResult(
        entities=[CanonicalEntity(...)],
        relations=[CanonicalRelation(...)],
        evidence=[CanonicalEvidence(...)],
        layer="L1",
        version="1.0"
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .evidence import Evidence
from .enums import (
    LayerType,
    OntologyRelationType,
    OntologyEntityType,
    CanonicalStatus,
    EntityType,
    RelationType,
    AttrValueType,
)


_STATUS_RANK: Dict[str, int] = {
    CanonicalStatus.RAW_CANDIDATE.value: 0,
    CanonicalStatus.STRUCTURED_CANDIDATE.value: 1,
    CanonicalStatus.SEMANTIC_CANDIDATE.value: 2,
    CanonicalStatus.INSTANCE_CANDIDATE.value: 3,
    CanonicalStatus.GOVERNED_CANONICAL.value: 4,
}

_LAYER_TO_STATUS: Dict[str, CanonicalStatus] = {
    LayerType.L0.value: CanonicalStatus.RAW_CANDIDATE,
    LayerType.L1.value: CanonicalStatus.STRUCTURED_CANDIDATE,
    LayerType.L2.value: CanonicalStatus.SEMANTIC_CANDIDATE,
    LayerType.L2_5.value: CanonicalStatus.INSTANCE_CANDIDATE,
    LayerType.L3.value: CanonicalStatus.INSTANCE_CANDIDATE,
}


def infer_status_from_layer(layer: str) -> CanonicalStatus:
    return _LAYER_TO_STATUS.get(layer, CanonicalStatus.RAW_CANDIDATE)


def can_transition(current: CanonicalStatus, target: CanonicalStatus) -> bool:
    return _STATUS_RANK[target.value] >= _STATUS_RANK[current.value]


class CanonicalStatusTransitionError(ValueError):
    """Raised when state machine attempts illegal backward transition."""


# ============================================================================
# Canonical Evidence - Unified evidence model (alias to Evidence)
# ============================================================================
CanonicalEvidence = Evidence

# ============================================================================
# Canonical Attribute - Unified attribute model
# ============================================================================

class CanonicalAttribute(BaseModel):
    attr_id: str
    name: str
    value: Any
    attr_type: str = "string"
    source: str = ""
    confidence: float = 1.0
    is_key: bool = False
    is_required: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator('attr_type')
    @classmethod
    def validate_attr_type(cls, v: str) -> str:
        valid_types = [t.value for t in AttrValueType]
        return v if v in valid_types else "string"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalAttribute":
        if not data:
            return cls(attr_id="", name="", value="")

        return cls(
            attr_id=data.get("attr_id", data.get("name", "")),
            name=data.get("name", ""),
            value=data.get("value", ""),
            attr_type=data.get("attr_type", "string"),
            source=data.get("source", ""),
            confidence=data.get("confidence", 1.0),
            is_key=data.get("is_key", False),
            is_required=data.get("is_required", False),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Entity Lineage - Entity lineage model
# ============================================================================

class EntityLineage(BaseModel):
    parent_entity_id: Optional[str] = None
    derived_from: List[str] = Field(default_factory=list)
    extracted_by: Optional[str] = None
    extraction_method: Optional[str] = None
    extraction_run_id: Optional[str] = None
    source_version: Optional[str] = None
    transformations: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    source_file: Optional[str] = None
    source_file_hash: Optional[str] = None
    extracted_at: Optional[str] = None
    pipeline_version: Optional[str] = None
    rule_version: Optional[str] = None
    model_version: Optional[str] = None


# ============================================================================
# Canonical Entity - Unified entity model
# ============================================================================

class CanonicalEntity(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    layer: str = "L1"
    status: str = CanonicalStatus.RAW_CANDIDATE.value
    description: str = ""
    aliases: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    attributes: List[CanonicalAttribute] = Field(default_factory=list)
    evidence: List[CanonicalEvidence] = Field(default_factory=list)
    source_trace: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    version: str = "1.0"
    lineage: EntityLineage = Field(default_factory=EntityLineage)
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator('layer')
    @classmethod
    def validate_layer(cls, v: str) -> str:
        valid_layers = [t.value for t in LayerType]
        return v if v in valid_layers else "L1"

    @field_validator('status', mode='before')
    @classmethod
    def validate_status(cls, v: Any) -> str:
        # Accept CanonicalStatus enum, valid strings; invalid values fallback to RAW_CANDIDATE
        if v is None or v == "":
            return CanonicalStatus.RAW_CANDIDATE.value
        value = v.value if isinstance(v, CanonicalStatus) else str(v)
        return value if value in _STATUS_RANK else CanonicalStatus.RAW_CANDIDATE.value

    @field_validator('created_at', 'updated_at', mode='before')
    @classmethod
    def set_timestamps(cls, v: str) -> str:
        return v if v else datetime.now().isoformat()

    @field_validator('attributes', mode='before')
    @classmethod
    def validate_attributes(cls, v: Any) -> List[CanonicalAttribute]:
        if not isinstance(v, list):
            return []
        return [CanonicalAttribute.from_dict(a) if isinstance(a, dict) else a for a in v]

    @field_validator('evidence', mode='before')
    @classmethod
    def validate_evidence(cls, v: Any) -> List[CanonicalEvidence]:
        if not isinstance(v, list):
            return []
        return [CanonicalEvidence.from_evidence_dict(e) if isinstance(e, dict) else e for e in v]

    def transition_to(self, target: CanonicalStatus) -> None:
        """Monotonic forward state transition; raises CanonicalStatusTransitionError on backward attempt."""
        current = CanonicalStatus(self.status)
        if not can_transition(current, target):
            raise CanonicalStatusTransitionError(
                f"illegal status transition: {current.value} -> {target.value}"
            )
        self.status = target.value
        self.updated_at = datetime.now().isoformat()

    @classmethod
    def from_ontology_entity(cls, entity: Any) -> "CanonicalEntity":
        evidence_list = []
        if hasattr(entity, 'evidence') and entity.evidence:
            if isinstance(entity.evidence, list):
                for e in entity.evidence:
                    evidence_list.append(CanonicalEvidence.from_evidence_dict(
                        e.to_dict() if hasattr(e, 'to_dict') else e))
            else:
                evidence_list.append(CanonicalEvidence.from_evidence_dict(
                    entity.evidence.to_dict() if hasattr(entity.evidence, 'to_dict') else entity.evidence))

        return cls(
            entity_id=entity.meta_id,
            name=entity.meta_name,
            entity_type=entity.meta_class,
            description=entity.description,
            properties=entity.properties,
            evidence=evidence_list,
            confidence=entity.confidence,
            version=entity.version,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalEntity":
        if not data:
            return cls(entity_id="", name="", entity_type="")

        layer = data.get("layer", "L1")
        # If dict doesn't explicitly provide status, infer initial state from layer
        status_raw = data.get("status")
        if status_raw is None or status_raw == "":
            status_raw = infer_status_from_layer(layer).value

        return cls(
            entity_id=data.get("entity_id", data.get("meta_id", "")),
            name=data.get("name", data.get("meta_name", "")),
            entity_type=data.get("entity_type", data.get("meta_class", "")),
            layer=layer,
            status=status_raw,
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            properties=data.get("properties", {}),
            attributes=data.get("attributes", []),
            evidence=data.get("evidence", []),
            source_trace=data.get("source_trace", {}),
            confidence=data.get("confidence", 1.0),
            version=str(data.get("version", "1.0")),
            lineage=data.get("lineage", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {}),
        )

    def add_attribute(self, name: str, value: Any, attr_type: str = "string",
                      source: str = "", confidence: float = 1.0) -> None:
        self.attributes.append(CanonicalAttribute(
            attr_id=f"{self.entity_id}-attr-{name}",
            name=name,
            value=value,
            attr_type=attr_type,
            source=source,
            confidence=confidence,
        ))

    def add_evidence(self, **kwargs) -> None:
        evidence_id = kwargs.pop('evidence_id', f"ev-{self.entity_id}-{len(self.evidence)}")
        self.evidence.append(CanonicalEvidence(evidence_id=evidence_id, **kwargs))


# ============================================================================
# Canonical Relation - Unified relation model
# ============================================================================

class CanonicalRelation(BaseModel):
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    description: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[CanonicalEvidence] = Field(default_factory=list)
    confidence: float = 1.0
    is_directed: bool = True
    weight: float = 1.0
    layer: str = "L1"
    status: str = CanonicalStatus.RAW_CANDIDATE.value
    created_at: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_file: Optional[str] = None
    source_file_hash: Optional[str] = None
    extracted_at: Optional[str] = None
    extraction_run_id: Optional[str] = None
    pipeline_version: Optional[str] = None
    rule_version: Optional[str] = None
    model_version: Optional[str] = None

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator('layer')
    @classmethod
    def validate_layer(cls, v: str) -> str:
        valid_layers = [t.value for t in LayerType]
        return v if v in valid_layers else "L1"

    @field_validator('status', mode='before')
    @classmethod
    def validate_status(cls, v: Any) -> str:
        if v is None or v == "":
            return CanonicalStatus.RAW_CANDIDATE.value
        value = v.value if isinstance(v, CanonicalStatus) else str(v)
        return value if value in _STATUS_RANK else CanonicalStatus.RAW_CANDIDATE.value

    @field_validator('created_at', mode='before')
    @classmethod
    def set_created_at(cls, v: str) -> str:
        return v if v else datetime.now().isoformat()

    @field_validator('evidence', mode='before')
    @classmethod
    def validate_evidence(cls, v: Any) -> List[CanonicalEvidence]:
        if not isinstance(v, list):
            return []
        return [CanonicalEvidence.from_evidence_dict(e) if isinstance(e, dict) else e for e in v]

    def transition_to(self, target: CanonicalStatus) -> None:
        """Monotonic forward state transition; raises CanonicalStatusTransitionError on backward attempt."""
        current = CanonicalStatus(self.status)
        if not can_transition(current, target):
            raise CanonicalStatusTransitionError(
                f"illegal status transition: {current.value} -> {target.value}"
            )
        self.status = target.value

    @classmethod
    def from_ontology_relationship(cls, rel: Any) -> "CanonicalRelation":
        evidence_list = []
        if hasattr(rel, 'evidence') and rel.evidence:
            evidence_list.append(CanonicalEvidence.from_evidence_dict(
                rel.evidence.to_dict() if hasattr(rel.evidence, 'to_dict') else rel.evidence))

        return cls(
            relation_id=rel.relationship_id,
            source_id=rel.source_id,
            target_id=rel.target_id,
            relation_type=rel.relationship_type,
            description=rel.description,
            properties=rel.properties,
            evidence=evidence_list,
            confidence=rel.confidence,
            layer=rel.layer,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalRelation":
        if not data:
            return cls(relation_id="", source_id="", target_id="", relation_type="")

        layer = data.get("layer", "L1")
        status_raw = data.get("status")
        if status_raw is None or status_raw == "":
            status_raw = infer_status_from_layer(layer).value

        return cls(
            relation_id=data.get("relation_id", data.get("relationship_id", "")),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relation_type=data.get("relation_type", data.get("relationship_type", "")),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            evidence=data.get("evidence", []),
            confidence=data.get("confidence", 1.0),
            is_directed=data.get("is_directed", True),
            weight=data.get("weight", 1.0),
            layer=layer,
            status=status_raw,
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
            source_file=data.get("source_file"),
            source_file_hash=data.get("source_file_hash"),
            extracted_at=data.get("extracted_at"),
            extraction_run_id=data.get("extraction_run_id"),
            pipeline_version=data.get("pipeline_version"),
            rule_version=data.get("rule_version"),
            model_version=data.get("model_version"),
        )


# ============================================================================
# Canonical Summary - Unified summary model
# ============================================================================

class CanonicalSummary(BaseModel):
    summary_id: str
    title: str
    content: str = ""
    key_points: List[str] = Field(default_factory=list)
    layer: str = "L1"
    version: str = "1.0"
    source_files: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    created_at: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator('created_at', mode='before')
    @classmethod
    def set_created_at(cls, v: str) -> str:
        return v if v else datetime.now().isoformat()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalSummary":
        if not data:
            return cls(summary_id="", title="")

        return cls(
            summary_id=data.get("summary_id", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            key_points=data.get("key_points", []),
            layer=data.get("layer", "L1"),
            version=data.get("version", "1.0"),
            source_files=data.get("source_files", []),
            confidence=data.get("confidence", 1.0),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Canonical Result - Unified result wrapper
# ============================================================================

class CanonicalResult(BaseModel):
    entities: List[CanonicalEntity] = Field(default_factory=list)
    relations: List[CanonicalRelation] = Field(default_factory=list)
    attributes: List[CanonicalAttribute] = Field(default_factory=list)
    evidence: List[CanonicalEvidence] = Field(default_factory=list)
    summaries: List[CanonicalSummary] = Field(default_factory=list)
    layer: str = "L1"
    version: str = "1.0"
    run_id: str = ""
    prompt_context: Dict[str, Any] = Field(default_factory=dict)
    source_files: List[str] = Field(default_factory=list)
    created_at: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('created_at', mode='before')
    @classmethod
    def set_created_at(cls, v: str) -> str:
        return v if v else datetime.now().isoformat()

    @field_validator('entities', mode='before')
    @classmethod
    def validate_entities(cls, v: Any) -> List[CanonicalEntity]:
        if not isinstance(v, list):
            return []
        return [CanonicalEntity.from_dict(e) if isinstance(e, dict) else e for e in v]

    @field_validator('relations', mode='before')
    @classmethod
    def validate_relations(cls, v: Any) -> List[CanonicalRelation]:
        if not isinstance(v, list):
            return []
        return [CanonicalRelation.from_dict(r) if isinstance(r, dict) else r for r in v]

    @field_validator('evidence', mode='before')
    @classmethod
    def validate_evidence(cls, v: Any) -> List[CanonicalEvidence]:
        if not isinstance(v, list):
            return []
        return [CanonicalEvidence.from_evidence_dict(e) if isinstance(e, dict) else e for e in v]

    @field_validator('summaries', mode='before')
    @classmethod
    def validate_summaries(cls, v: Any) -> List[CanonicalSummary]:
        if not isinstance(v, list):
            return []
        return [CanonicalSummary.from_dict(s) if isinstance(s, dict) else s for s in v]

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    @property
    def avg_confidence(self) -> float:
        if not self.entities:
            return 0.0
        return sum(e.confidence for e in self.entities) / len(self.entities)

    @property
    def source_file_count(self) -> int:
        return len(self.source_files)

    def add_entity(self, entity: CanonicalEntity) -> None:
        self.entities.append(entity)

    def add_relation(self, relation: CanonicalRelation) -> None:
        self.relations.append(relation)

    def add_evidence(self, evidence: CanonicalEvidence) -> None:
        self.evidence.append(evidence)

    def add_summary(self, summary: CanonicalSummary) -> None:
        self.summaries.append(summary)

    def merge(self, other: "CanonicalResult") -> None:
        self.entities.extend(other.entities)
        self.relations.extend(other.relations)
        self.attributes.extend(other.attributes)
        self.evidence.extend(other.evidence)
        self.summaries.extend(other.summaries)
        self.source_files.extend(other.source_files)
        self.prompt_context.update(other.prompt_context)
        self.metadata.update(other.metadata)

    def deduplicate(self) -> "CanonicalResult":
        seen_entity_ids = set()
        seen_relation_ids = set()

        deduped_entities = []
        for entity in self.entities:
            if entity.entity_id and entity.entity_id not in seen_entity_ids:
                seen_entity_ids.add(entity.entity_id)
                deduped_entities.append(entity)

        deduped_relations = []
        for relation in self.relations:
            if relation.relation_id and relation.relation_id not in seen_relation_ids:
                seen_relation_ids.add(relation.relation_id)
                deduped_relations.append(relation)

        return CanonicalResult(
            entities=deduped_entities,
            relations=deduped_relations,
            attributes=self.attributes,
            evidence=self.evidence,
            summaries=self.summaries,
            layer=self.layer,
            version=self.version,
            run_id=self.run_id,
            prompt_context=self.prompt_context,
            source_files=self.source_files,
            metadata=self.metadata,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalResult":
        if not data:
            return cls()

        return cls(
            entities=data.get("entities", []),
            relations=data.get("relations", []),
            evidence=data.get("evidence", []),
            summaries=data.get("summaries", []),
            layer=data.get("layer", "L1"),
            version=data.get("version", "1.0"),
            run_id=data.get("run_id", ""),
            prompt_context=data.get("prompt_context", {}),
            source_files=data.get("source_files", []),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = self.model_dump()
        result["_stats"] = {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "avg_confidence": self.avg_confidence,
            "source_file_count": self.source_file_count,
        }
        return result


__all__ = [
    "LayerType",
    "EntityType",
    "RelationType",
    "AttrValueType",
    "CanonicalEvidence",
    "CanonicalAttribute",
    "EntityLineage",
    "CanonicalEntity",
    "CanonicalRelation",
    "CanonicalSummary",
    "CanonicalResult",
]
