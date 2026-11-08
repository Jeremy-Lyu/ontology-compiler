"""Unified Fragment model for ontology extraction outputs.

This module is the canonical source for Fragment models.
Shared between src and ontology_compiler to avoid duplication.

[LT01 2026-06-14]: Upgraded to Pydantic BaseModel for automatic serialization.

This module provides a standardized Fragment model to ensure consistency
across different extraction paths (TBH/RIGOR vs folder analysis).

Benefits of Pydantic upgrade:
- Auto-generated serialization (model_dump())
- Auto-generated deserialization (model_validate())
- Built-in validation
- Schema generation
- Reduced boilerplate (~1000 lines simplified)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .evidence import Evidence


class FragmentNode(BaseModel):
    """Represents a node/concept extracted from a document.

    Unified model that consolidates both 'classes' (domain concepts)
    and 'nodes' (instance concepts) from different extraction paths.
    """

    meta_id: str
    meta_name: str
    meta_class: str = "Concept"
    domain: str = "SUPPLY_CHAIN"
    source_file: str = ""
    description: str = ""
    layer: str = ""
    layer_scope: str = "spec"
    status: str = "active"
    entity_type: str = ""
    sub_type: str = ""
    classification_confidence: float = 0.0
    needs_review: bool = False
    evidence: Optional[Evidence] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('properties', mode='before')
    @classmethod
    def validate_properties(cls, v: Any) -> Dict[str, Any]:
        """Ensure properties is a dict."""
        return v if isinstance(v, dict) else {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FragmentNode":
        """Create from dictionary representation (legacy compatibility)."""
        return cls.model_validate(data)


class FragmentInstance(BaseModel):
    """Represents a document-level instance extracted from source."""

    meta_id: str
    meta_name: str
    meta_class: str = "Instance"
    domain: str = "SUPPLY_CHAIN"
    source_file: str = ""
    description: str = ""
    layer: str = ""
    layer_scope: str = "spec"
    status: str = "active"
    entity_type: str = ""
    sub_type: str = ""
    classification_confidence: float = 0.0
    needs_review: bool = False
    evidence: Optional[Evidence] = None
    values: Dict[str, Any] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('values', mode='before')
    @classmethod
    def validate_values(cls, v: Any) -> Dict[str, Any]:
        """Ensure values is a dict."""
        return v if isinstance(v, dict) else {}

    @field_validator('properties', mode='before')
    @classmethod
    def validate_properties(cls, v: Any) -> Dict[str, Any]:
        """Ensure properties is a dict."""
        return v if isinstance(v, dict) else {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FragmentInstance":
        """Create from dictionary representation (legacy compatibility)."""
        return cls.model_validate(data)


class FragmentRelationship(BaseModel):
    """Represents a relationship between entities.

    Canonical fields: source, target, relation_type
    Legacy aliases accepted at input: from->source, to->target, type/relation_type->relation_type
    """

    source: str = ""
    target: str = ""
    relation_type: str = ""
    type: str = ""
    domain: str = "SUPPLY_CHAIN"
    source_file: str = ""
    evidence_ref: str = ""
    citation: str = ""
    rule_no: str = ""
    confidence: float = 0.0
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('properties', mode='before')
    @classmethod
    def validate_properties(cls, v: Any) -> Dict[str, Any]:
        """Ensure properties is a dict."""
        return v if isinstance(v, dict) else {}

    @model_validator(mode='before')
    @classmethod
    def normalize_fields(cls, values: Any) -> Any:
        """Normalize legacy field names to canonical names."""
        if isinstance(values, dict):
            if not values.get('source'):
                values['source'] = values.pop('from', '') if 'from' in values else values.get('source', '')
            if not values.get('target'):
                values['target'] = values.pop('to', '') if 'to' in values else values.get('target', '')
            rt = values.get('relation_type', '') or values.get('type', '') or values.get('rel_type', '') or values.get('relation', '')
            if rt:
                values['relation_type'] = rt
                values['type'] = rt
        return values

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FragmentRelationship":
        """Create from dictionary representation (legacy compatibility)."""
        return cls.model_validate(data)


class Fragment(BaseModel):
    """Unified Fragment model for ontology extraction outputs.

    This model standardizes the Fragment structure across all extraction paths:
    - TBH/RIGOR path: uses 'classes' + 'nodes' + 'relationships'
    - Folder analysis path: uses 'nodes' + 'edges'

    The unified model uses:
    - nodes: List[FragmentNode] - all concepts (both domain and instance-level)
    - instances: List[FragmentInstance] - document-level instances
    - relationships: List[FragmentRelationship] - all relationships (canonical name)
    """

    source_file: str = ""
    domain: str = "SUPPLY_CHAIN"
    schema_version: str = "1.0"
    nodes: List[FragmentNode] = Field(default_factory=list)
    instances: List[FragmentInstance] = Field(default_factory=list)
    relationships: List[FragmentRelationship] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('nodes', mode='before')
    @classmethod
    def validate_nodes(cls, v: Any) -> List[FragmentNode]:
        """Ensure nodes is a list of valid FragmentNodes."""
        if not isinstance(v, list):
            return []
        return [FragmentNode.model_validate(n) for n in v]

    @field_validator('instances', mode='before')
    @classmethod
    def validate_instances(cls, v: Any) -> List[FragmentInstance]:
        """Ensure instances is a list of valid FragmentInstances."""
        if not isinstance(v, list):
            return []
        return [FragmentInstance.model_validate(i) for i in v]

    @field_validator('relationships', mode='before')
    @classmethod
    def validate_relationships(cls, v: Any) -> List[FragmentRelationship]:
        """Ensure relationships is a list of valid FragmentRelationships."""
        if not isinstance(v, list):
            return []
        return [FragmentRelationship.model_validate(r) for r in v]

    @field_validator('metadata', mode='before')
    @classmethod
    def validate_metadata(cls, v: Any) -> Dict[str, Any]:
        """Ensure metadata is a dict."""
        return v if isinstance(v, dict) else {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (standardized format)."""
        result = self.model_dump()
        result["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            **result.get("metadata", {}),
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fragment":
        """Create from dictionary representation (handles both old and new formats)."""
        # Handle relationships (may come from 'relationships' or 'edges')
        if "edges" in data and "relationships" not in data:
            data = dict(data)
            data["relationships"] = data.pop("edges")

        return cls.model_validate(data)

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any]) -> "Fragment":
        """Create from legacy dictionary format (handles field name variations).

        This method specifically handles the inconsistencies between:
        - RIGOR path: uses 'classes' for domain concepts
        - Folder analysis path: uses 'nodes' for all concepts
        - Mixed usage of 'relationships' vs 'edges'
        """
        source_file = data.get("source_file", data.get("metadata", {}).get("source_file", ""))
        domain = data.get("domain", "SUPPLY_CHAIN")

        # Collect nodes from all possible sources
        nodes: List[FragmentNode] = []
        seen_ids: set = set()

        # From 'classes' (RIGOR domain concepts)
        for cls_data in data.get("classes", []):
            if isinstance(cls_data, dict):
                name = cls_data.get("name") or cls_data.get("meta_name", "")
                if name:
                    node_id = cls_data.get("meta_id") or f"CLS-{name.upper().replace(' ', '_')}"
                    if node_id not in seen_ids:
                        seen_ids.add(node_id)
                        evidence_data = cls_data.get("evidence")
                        evidence = Evidence.model_validate(evidence_data) if evidence_data else None
                        nodes.append(FragmentNode(
                            meta_id=node_id,
                            meta_name=name,
                            meta_class=cls_data.get("meta_class", "Concept"),
                            domain=domain,
                            source_file=source_file,
                            description=cls_data.get("description", ""),
                            evidence=evidence,
                            properties={k: v for k, v in cls_data.items()
                                        if k not in {"meta_id", "meta_name", "meta_class", "domain", "source_file", "description", "name", "evidence"}}
                        ))

        # From 'nodes'
        for node_data in data.get("nodes", []):
            if isinstance(node_data, dict):
                node_id = node_data.get("meta_id", "")
                if node_id and node_id not in seen_ids:
                    seen_ids.add(node_id)
                    node_data_copy = dict(node_data)
                    if "meta_name" not in node_data_copy:
                        node_data_copy["meta_name"] = node_id
                    nodes.append(FragmentNode.model_validate(node_data_copy))

        # Handle relationships (may be 'relationships' or 'edges')
        rels_data = data.get("relationships", data.get("edges", []))
        relationships = []
        for rel_data in rels_data:
            if isinstance(rel_data, dict):
                rel_data_copy = dict(rel_data)
                if "type" not in rel_data_copy:
                    rel_data_copy["type"] = "RELATED_TO"
                relationships.append(FragmentRelationship.model_validate(rel_data_copy))

        return cls(
            source_file=source_file,
            domain=domain,
            nodes=nodes,
            instances=[],
            relationships=relationships,
            metadata=data.get("metadata", {})
        )

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the fragment."""
        return {
            "total_entities": len(self.nodes) + len(self.instances),
            "nodes": len(self.nodes),
            "instances": len(self.instances),
            "relationships": len(self.relationships),
        }

    def merge(self, other: "Fragment") -> "Fragment":
        """Merge another fragment into this one and return the merged result.
        
        Args:
            other: Fragment to merge into this one
            
        Returns:
            New merged Fragment instance
        """
        merged = Fragment(
            source_file=self.source_file,
            domain=self.domain,
            schema_version=self.schema_version,
            nodes=self.nodes.copy(),
            instances=self.instances.copy(),
            relationships=self.relationships.copy(),
            metadata=self.metadata.copy(),
        )

        seen_node_ids = {node.meta_id for node in merged.nodes}
        for node in other.nodes:
            if node.meta_id not in seen_node_ids:
                seen_node_ids.add(node.meta_id)
                merged.nodes.append(node)

        seen_instance_ids = {inst.meta_id for inst in merged.instances}
        for inst in other.instances:
            if inst.meta_id not in seen_instance_ids:
                seen_instance_ids.add(inst.meta_id)
                merged.instances.append(inst)

        seen_rels = set()
        for rel in merged.relationships:
            seen_rels.add((rel.source, rel.target, rel.type))
        for rel in other.relationships:
            key = (rel.source, rel.target, rel.type)
            if key not in seen_rels:
                seen_rels.add(key)
                merged.relationships.append(rel)

        return merged
