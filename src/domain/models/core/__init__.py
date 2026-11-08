"""Core domain models and shared governance contracts."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..ontology_entity import OntologyEntity
from ..ontology_relationship import OntologyRelationship
from ..evidence import Evidence
from ..canonical import (
    CanonicalAttribute,
    CanonicalEvidence,
    EntityLineage,
    CanonicalEntity,
    CanonicalRelation,
    CanonicalSummary,
    CanonicalResult,
)


def _now_iso() -> str:
    return datetime.now().isoformat()


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class TraceContext(ContractModel):
    contract_name: str = "trace_context"
    contract_version: str = "1.0.0"
    trace_id: str = ""
    session_id: str = ""
    run_id: str = ""
    stage_id: str = ""
    task_id: str = ""
    artifact_id: str = ""
    document_id: str = ""
    chunk_id: str = ""
    prompt_id: str = ""
    skill_id: str = ""
    model_call_id: str = ""
    parent_trace_id: str = ""
    created_at: str = Field(default_factory=_now_iso)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactEnvelope(ContractModel):
    contract_name: str = "artifact_envelope"
    contract_version: str = "1.0.0"
    artifact_id: str = ""
    artifact_type: str = ""
    artifact_version: str = "1.0.0"
    status: str = "draft"
    trace_context: TraceContext = Field(default_factory=TraceContext)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    quality: Dict[str, Any] = Field(default_factory=dict)
    payload: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentRef(ContractModel):
    document_id: str = ""
    document_type: str = ""
    source_path: str = ""
    hash: str = ""
    title: str = ""
    language: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkRef(ContractModel):
    chunk_id: str = ""
    document_id: str = ""
    start_offset: int = 0
    end_offset: int = 0
    evidence_span: str = ""
    chunk_hash: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityRef(ContractModel):
    entity_id: str = ""
    canonical_name: str = ""
    aliases: List[str] = Field(default_factory=list)
    entity_type: str = ""
    layer: str = ""
    confidence: float = 0.0
    validation_state: str = "pending"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, value))


class RelationRef(ContractModel):
    relation_id: str = ""
    source_id: str = ""
    target_id: str = ""
    relation_type: str = ""
    confidence: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)
    validation_state: str = "pending"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, value))


class EvidenceRef(ContractModel):
    evidence_id: str = ""
    source_document_id: str = ""
    source_chunk_id: str = ""
    span: str = ""
    support_level: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, value))


class GraphPath(ContractModel):
    path_id: str = ""
    nodes: List[str] = Field(default_factory=list)
    relations: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalAssetPacket(ContractModel):
    contract_name: str = "retrieval_asset_packet"
    contract_version: str = "1.0.0"
    packet_id: str = ""
    ontology_version: str = "1.0.0"
    run_id: str = ""
    trace_context: TraceContext = Field(default_factory=TraceContext)
    document_refs: List[DocumentRef] = Field(default_factory=list)
    chunk_refs: List[ChunkRef] = Field(default_factory=list)
    entity_refs: List[EntityRef] = Field(default_factory=list)
    relation_refs: List[RelationRef] = Field(default_factory=list)
    evidence_refs: List[EvidenceRef] = Field(default_factory=list)
    graph_paths: List[GraphPath] = Field(default_factory=list)
    structured_filters: Dict[str, Any] = Field(default_factory=dict)
    retrieval_text: List[str] = Field(default_factory=list)
    quality: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)


class SkillDefinition(ContractModel):
    contract_name: str = "skill_definition"
    contract_version: str = "1.0.0"
    skill_id: str = ""
    name: str = ""
    version: str = "1.0.0"
    status: str = "stable"
    owner: str = ""
    description: str = ""
    command: str = ""
    script_path: str = ""
    category: str = ""
    enabled: bool = True
    capabilities: List[str] = Field(default_factory=list)
    business_domain: List[str] = Field(default_factory=list)
    layer_scope: List[str] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    side_effects: List[str] = Field(default_factory=list)
    determinism: str = "deterministic"
    idempotent: bool = True
    required_evidence: bool = True
    fallback_skills: List[str] = Field(default_factory=list)
    compatible_pipeline_stages: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    quality_thresholds: Dict[str, Any] = Field(default_factory=dict)
    trace_config: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    deprecation: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineStageResult(ContractModel):
    stage: str
    status: str
    duration_ms: int = 0
    command: Optional[List[str]] = None
    warnings: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error_type: Optional[str] = None
    action_hint: Optional[str] = None
    dispatch: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class PipelineExecutionResult(ContractModel):
    contract_name: str = "pipeline_execution_result"
    contract_version: str = "1.0.0"
    run_id: str = ""
    pipeline_name: str = ""
    pipeline_version: str = "1.0.0"
    overall_status: str = "SUCCESS"
    execution_mode: str = "production"
    stages: List[PipelineStageResult] = Field(default_factory=list)
    source_path: str = ""
    domain: str = "AUTO"
    entity_count: int = 0
    relation_count: int = 0
    elapsed_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    trace_context: TraceContext = Field(default_factory=TraceContext)
    artifact_refs: List[str] = Field(default_factory=list)
    validation: Dict[str, Any] = Field(default_factory=dict)
    cache_stats: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_summary_payload(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "results": [stage.model_dump(exclude_none=True) for stage in self.stages],
        }


class ValidationCheck(ContractModel):
    check_id: str = ""
    name: str = ""
    status: str = "passed"
    message: str = ""
    severity: str = "info"
    details: Dict[str, Any] = Field(default_factory=dict)


class ValidationReport(ContractModel):
    contract_name: str = "validation_report"
    contract_version: str = "1.0.0"
    report_id: str = ""
    target_ref: str = ""
    target_type: str = ""
    status: str = "passed"
    validation_state: str = "passed"
    execution_mode: str = "production"
    checks: List[ValidationCheck] = Field(default_factory=list)
    summary: Dict[str, int] = Field(default_factory=dict)
    trace_context: TraceContext = Field(default_factory=TraceContext)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "OntologyEntity",
    "OntologyRelationship",
    "Evidence",
    "CanonicalAttribute",
    "CanonicalEvidence",
    "EntityLineage",
    "CanonicalEntity",
    "CanonicalRelation",
    "CanonicalSummary",
    "CanonicalResult",
    "TraceContext",
    "ArtifactEnvelope",
    "DocumentRef",
    "ChunkRef",
    "EntityRef",
    "RelationRef",
    "EvidenceRef",
    "GraphPath",
    "RetrievalAssetPacket",
    "SkillDefinition",
    "PipelineStageResult",
    "PipelineExecutionResult",
    "ValidationCheck",
    "ValidationReport",
]
