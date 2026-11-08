"""Skill contract models.

These contracts capture the T7 three-layer skill model in code so skills can be
versioned and orchestrated consistently.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class SkillRuntimeType(str, Enum):
    PYTHON = "python"
    PROMPT = "prompt"
    HYBRID = "hybrid"
    EXTERNAL = "external"


class SkillStatus(str, Enum):
    STABLE = "stable"
    MATURING = "maturing"
    DEPRECATED = "deprecated"


class SkillContractName(str, Enum):
    SKILL_DEFINITION = "skill_definition"


class SkillDeterminism(str, Enum):
    DETERMINISTIC = "deterministic"
    NON_DETERMINISTIC = "non_deterministic"


class SkillDefinition(BaseModel):
    """Canonical definition shared by skill registry and pipeline routing."""

    contract_name: SkillContractName = Field(SkillContractName.SKILL_DEFINITION, description="Contract discriminator")
    contract_version: str = Field("1.0.0", description="Contract version")
    skill_id: Optional[str] = Field(None, description="Stable skill identifier")
    name: str = Field(..., description="Human-readable skill name")
    version: str = Field(..., description="Skill version")
    description: str = Field(..., description="What the skill does")
    status: SkillStatus = Field(SkillStatus.MATURING, description="Lifecycle status")
    owner: Optional[str] = None
    command: Optional[str] = None
    script_path: Optional[str] = None
    category: Optional[str] = None
    enabled: bool = True
    capabilities: List[str] = Field(default_factory=list)
    business_domain: List[str] = Field(default_factory=list)
    layer_scope: List[str] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    side_effects: List[str] = Field(default_factory=list)
    determinism: SkillDeterminism = SkillDeterminism.DETERMINISTIC
    idempotent: Optional[bool] = None
    required_evidence: bool = False
    fallback_skills: List[str] = Field(default_factory=list)
    compatible_pipeline_stages: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    quality_thresholds: Dict[str, Any] = Field(default_factory=dict)
    trace_config: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    deprecation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _default_skill_id(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            value.setdefault("skill_id", value.get("name"))
        return value


class SkillSpec(BaseModel):
    skill_name: str = Field(..., description="Stable skill identifier")
    version: str = Field("v1", description="Skill version")
    purpose: str = Field(..., description="What the skill is for")
    inputs: List[str] = Field(default_factory=list, description="Expected inputs")
    outputs: List[str] = Field(default_factory=list, description="Expected outputs")
    constraints: List[str] = Field(default_factory=list, description="Runtime constraints")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies")
    quality_expectations: List[str] = Field(default_factory=list, description="Quality requirements")
    fallback_behavior: str = Field("conservative", description="Fallback behavior")
    observability_fields: List[str] = Field(default_factory=list, description="Trace fields")
    runtime_type: SkillRuntimeType = Field(SkillRuntimeType.HYBRID, description="Implementation style")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class SkillRuntimeContract(BaseModel):
    skill_name: str = Field(..., description="Stable skill identifier")
    spec_version: str = Field("v1", description="Bound spec version")
    runtime_type: SkillRuntimeType = Field(SkillRuntimeType.HYBRID, description="Runtime type")
    allowed_calls: List[str] = Field(default_factory=list, description="Allowed downstream calls")
    failure_policy: str = Field("conservative", description="Failure policy")
    fallback_policy: str = Field("fallback-to-minimal", description="Fallback policy")
    replayability: bool = Field(True, description="Whether runs should be replayable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


__all__ = [
    "SkillRuntimeType",
    "SkillStatus",
    "SkillContractName",
    "SkillDeterminism",
    "SkillDefinition",
    "SkillSpec",
    "SkillRuntimeContract",
]
