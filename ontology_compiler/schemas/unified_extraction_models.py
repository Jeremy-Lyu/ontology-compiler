"""
Pydantic Models for UnifiedExtractionResult validation.

Phase 1 Improvement: Structured validation with Pydantic v2.
Provides type-safe, performant validation for LLM extraction output.
"""

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator


class EntityAttribute(BaseModel):
    """Entity attribute with name-value pair."""
    name: str = Field(..., min_length=1, description="Attribute name")
    value: str = Field(default="", description="Attribute value")


class L1Entity(BaseModel):
    """L1 Domain Concept entity."""
    meta_id: str = Field(..., min_length=1, description="Unique entity ID (e.g., L1-001)")
    meta_name: str = Field(..., min_length=1, description="Entity name")
    meta_class: str = Field(default="Concept", min_length=1, description="Entity class")
    meta_subclass: Optional[str] = Field(None, description="Entity subclass")
    description: Optional[str] = Field(None, description="Entity description")
    domain: Optional[str] = Field(None, description="Domain this entity belongs to")
    evidence: Union[str, Dict[str, Any]] = Field(default="Extracted from document content", description="Evidence from source document (string or structured dict)")
    attributes: Optional[List[EntityAttribute]] = Field(default_factory=list, description="Additional attributes")

    @field_validator('meta_id')
    @classmethod
    def validate_l1_meta_id(cls, v: str) -> str:
        if not v.startswith(('L1-', 'L1_')):
            raise ValueError(f"L1 entity meta_id must start with L1- or L1_, got: {v}")
        return v

    @field_validator('attributes', mode='before')
    @classmethod
    def normalize_attributes(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            return [EntityAttribute(name=str(k), value=str(val)) for k, val in v.items()]
        if isinstance(v, list):
            normalized: List[EntityAttribute] = []
            for item in v:
                if isinstance(item, EntityAttribute):
                    normalized.append(item)
                elif isinstance(item, str):
                    normalized.append(EntityAttribute(name=item, value=""))
                elif isinstance(item, dict):
                    name = str(item.get("name") or item.get("key") or item.get("attribute") or "").strip()
                    if not name:
                        for key, value in item.items():
                            if key != "value":
                                name = str(key).strip()
                                if name:
                                    normalized.append(EntityAttribute(name=name, value=str(value)))
                                break
                        continue
                    value = item.get("value")
                    normalized.append(EntityAttribute(name=name, value="" if value is None else str(value)))
                else:
                    normalized.append(EntityAttribute(name=str(item), value=""))
            return normalized
        return []


class L2Entity(BaseModel):
    """L2 Business Scenario entity."""
    meta_id: str = Field(..., min_length=1, description="Unique scenario ID (e.g., L2-001)")
    scenario_name: str = Field(..., min_length=1, description="Scenario name")
    domain: Optional[str] = Field(None, description="Domain this scenario belongs to")
    description: Optional[str] = Field(None, description="Scenario description")
    evidence: Union[str, Dict[str, Any]] = Field(default="Extracted from document content", description="Evidence from source document (string or structured dict)")
    participants: Optional[List[str]] = Field(default_factory=list, description="Entities participating in scenario")
    triggers: Optional[List[str]] = Field(default_factory=list, description="Events that trigger this scenario")
    outcomes: Optional[List[str]] = Field(default_factory=list, description="Expected outcomes")
    preconditions: Optional[List[str]] = Field(default_factory=list, description="Preconditions for scenario")
    steps: Optional[List[str]] = Field(default_factory=list, description="Scenario steps")

    @field_validator('meta_id')
    @classmethod
    def validate_l2_meta_id(cls, v: str) -> str:
        if not v.startswith(('L2-', 'L2_')):
            raise ValueError(f"L2 entity meta_id must start with L2- or L2_, got: {v}")
        return v


class L25Entity(BaseModel):
    """L2.5 Mapping Rule entity."""
    meta_id: Optional[str] = Field(None, description="Unique rule ID (e.g., L25-001)")
    name: Optional[str] = Field(None, description="Rule name")
    rule_type: Optional[str] = Field(default="mapping", description="Type of rule")
    source_entity: Optional[str] = Field(None, description="Source entity ID")
    target_entity: Optional[str] = Field(None, description="Target entity ID")
    transformation_logic: Optional[str] = Field(None, description="Transformation logic")
    constraints: Optional[List[str]] = Field(default_factory=list, description="Rule constraints")
    validation_rules: Optional[List[str]] = Field(default_factory=list, description="Validation rules")
    description: Optional[str] = Field(None, description="Rule description")
    evidence: Union[str, Dict[str, Any]] = Field(default="Extracted from document content", description="Evidence from source document (string or structured dict)")


class Relationship(BaseModel):
    """Relationship between entities."""
    from_: str = Field(..., alias="from", min_length=1, description="Source entity ID")
    to: str = Field(..., min_length=1, description="Target entity ID")
    type: str = Field(..., min_length=1, description="Relationship type (e.g., SEMANTIC_LINK, PARTICIPATES_IN)")
    layer: Optional[str] = Field(None, description="Layer this relationship spans (e.g., L1_L1, L1_L2)")
    description: Optional[str] = Field(None, description="Relationship description")
    evidence_ref: Optional[str] = Field(None, description="Source citation supporting the link (e.g. doc:slide-5)")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional relationship properties")


class UnifiedExtractionData(BaseModel):
    """
    Complete extraction result with Pydantic validation.
    
    Phase 1 Improvement: Type-safe validation for all three layers.
    """
    l1_entities: List[L1Entity] = Field(default_factory=list, description="L1 domain concepts")
    l2_entities: List[L2Entity] = Field(default_factory=list, description="L2 business scenarios")
    l25_entities: List[L25Entity] = Field(default_factory=list, description="L2.5 mapping rules")
    relationships: List[Relationship] = Field(default_factory=list, description="Entity relationships")

    @field_validator('l1_entities')
    @classmethod
    def validate_l1_not_empty(cls, v: List[L1Entity]) -> List[L1Entity]:
        if len(v) == 0:
            # Warning only, not error - partial results are acceptable
            pass
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with existing code."""
        return {
            "l1_entities": [e.model_dump() for e in self.l1_entities],
            "l2_entities": [e.model_dump() for e in self.l2_entities],
            "l25_entities": [e.model_dump() for e in self.l25_entities],
            "relationships": [r.model_dump(by_alias=True) for r in self.relationships],
        }

    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return {
            "l1_count": len(self.l1_entities),
            "l2_count": len(self.l2_entities),
            "l25_count": len(self.l25_entities),
            "relationship_count": len(self.relationships),
        }


class ExtractionValidationResult(BaseModel):
    """Validation result for unified extraction payloads."""
    valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    stats: Optional[Dict[str, int]] = Field(None, description="Extraction statistics")

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# Schema version for tracking
SCHEMA_VERSION = "2.0.0-phase1"
