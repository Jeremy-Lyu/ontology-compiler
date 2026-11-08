from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from core.utils.relationship_helpers import entity_to_dict
from src.domain.models import Fragment, FragmentRelationship

# Core extraction
from ontology_compiler.agents.file_parser import FileParser
from ontology_compiler.agents.extractor_plugin import get_plugin
from ontology_compiler.services.extraction.true_unified_extractor import TrueUnifiedExtractor

# Quality and validation
from ontology_compiler.services.merge.entity_dedup_service import EntityDeduplicationService, LLMConflictResolver
from ontology_compiler.services.quality_assurance.pipelines import ConflictDetectionPipeline, QualityGatePipeline
from ontology_compiler.services.change_detector import ChangeDetector
from ontology_compiler.services.cross_file_association_service import CrossFileAssociationService, SemanticEquivalenceJudge

# [Phase 3 - TRACE 2026-06-17]: Execution tracer
from ontology_compiler.services.execution_tracer import get_tracer, ExecutionContext

# [VERSION-MIGRATION]: Schema registry for version management
from core.schema_registry import get_schema_registry

# [Phase 3 - SELF-EVOLVING LOOP]: Skill telemetry and evolution (archived)
try:
    from ontology_compiler.services.skill_telemetry import get_telemetry, SkillTelemetry
except ImportError:
    class SkillTelemetry: pass
    def get_telemetry(): return None

try:
    from ontology_compiler.services.skill_analyzer import get_analyzer, SkillAnalyzer
except ImportError:
    class SkillAnalyzer: pass
    def get_analyzer(): return None

try:
    from ontology_compiler.services.skill_proposal_engine import get_proposal_engine, SkillProposalEngine
except ImportError:
    class SkillProposalEngine: pass
    def get_proposal_engine(): return None

try:
    from ontology_compiler.services.skill_review_gate import get_review_gate, SkillReviewGate
except ImportError:
    class SkillReviewGate: pass
    def get_review_gate(): return None

try:
    from ontology_compiler.harness.skill_registry import SkillVersionManager
except ImportError:
    class SkillVersionManager: pass

logger = logging.getLogger(__name__)


@dataclass
class CompilationRunResult:
    success: bool
    run_id: str
    quality_level: str
    quality_score: float
    nodes: list[Any]
    instances: list[Any]
    relationships: list[Any]
    node_count: int
    edge_count: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status: str = "SUCCESS"
    publishable: bool = True

    @property
    def degraded(self) -> bool:
        return self.status == "DEGRADED_SUCCESS"

    @property
    def is_publishable(self) -> bool:
        return self.publishable and self.success and self.status == "SUCCESS"


def _canonicalize_entity(entity: Any) -> Any:
    if isinstance(entity, dict):
        entity = dict(entity)
        if entity.get("meta_name") and not entity.get("name"):
            entity["name"] = entity["meta_name"]
        if entity.get("meta_class") and not entity.get("entity_type"):
            entity["entity_type"] = entity["meta_class"]
        if entity.get("layer") == "L3":
            entity["entity_type"] = "Instance"
        return entity
    return entity.model_dump() if hasattr(entity, "model_dump") else dict(entity)


def _canonicalize_relationship(rel: Any) -> Any:
    if isinstance(rel, dict):
        rel = dict(rel)
        rel["source_id"] = rel.get("source_id") or rel.get("from") or rel.get("source")
        rel["target_id"] = rel.get("target_id") or rel.get("to") or rel.get("target")
        rel["relation_type"] = rel.get("relation_type") or rel.get("type") or rel.get("rel_type")
        return rel
    return rel.model_dump() if hasattr(rel, "model_dump") else dict(rel)


def _extract_error_messages(metadata: Dict[str, Any], fallback: str) -> list[str]:
    errors = metadata.get("errors") if isinstance(metadata, dict) else None
    if isinstance(errors, list):
        return [str(item) for item in errors if item is not None]
    if errors:
        return [str(errors)]
    return [fallback]


def _has_meaningful_payload(result: Dict[str, Any]) -> bool:
    nodes = result.get("nodes") or []
    instances = result.get("instances") or []
    relationships = result.get("relationships") or []
    return bool(nodes or instances or relationships)


def degraded_result_warning(result: Dict[str, Any], metadata: Dict[str, Any] | None = None) -> str:
    """Name the actual failure. A partial graph is not an offline empty result."""
    metadata = metadata or {}
    meaningful = _has_meaningful_payload(result)
    offline = bool(metadata.get("offline_fallback")) or bool(result.get("offline_fallback"))
    if not meaningful and offline:
        return "offline fallback: extractor returned an empty result"
    if not meaningful:
        return "extractor returned an empty result"
    return "degraded extraction: partial result kept after validation or truncation"


def _normalize_status(raw_status: Any) -> str:
    status = str(raw_status or "").strip().upper()
    if status in {"SUCCESS", "DEGRADED_SUCCESS", "RETRYABLE_FAILURE", "CONTRACT_FAILURE", "FATAL_FAILURE", "FAILED"}:
        return status
    if status in {"PARTIAL", "WARN", "WARNING", "DEGRADED"}:
        return "DEGRADED_SUCCESS"
    return "SUCCESS"


def collect_entity_ids(entities: list[Any]) -> set[str]:
    ids: set[str] = set()
    for entity in entities:
        if isinstance(entity, dict):
            candidate = entity.get("meta_id") or entity.get("entity_id") or entity.get("id")
        else:
            candidate = getattr(entity, "meta_id", None) or getattr(entity, "entity_id", None) or getattr(entity, "id", None)
        if candidate:
            ids.add(str(candidate))
    return ids


def prune_dangling_relationships(relationships: list[Any], entity_ids: set[str], allow_dangling: bool = False):
    pruned = []
    dropped = 0
    for rel in relationships or []:
        rel_obj = _canonicalize_relationship(rel)
        source_id = rel_obj.get("source_id")
        target_id = rel_obj.get("target_id")
        if allow_dangling or ((not source_id or source_id in entity_ids) and (not target_id or target_id in entity_ids)):
            pruned.append(rel)
        else:
            dropped += 1
    return pruned, dropped


@dataclass
class CompilationRequest:
    """Backward-compatible request object for pipeline entrypoints."""

    source_path: str
    is_folder: bool = False
    run_id: str = ""
    parent_run_id: Optional[str] = None
    domain: str = "SUPPLY_CHAIN"
    skip_rigor: bool = False
    force_llm_extraction: bool = False
    fast_mode: bool = False
    max_files: Optional[int] = None
    output_dir: Optional[str] = None
    analysis_question: Optional[str] = None
    raw_text: str = ""
    use_agent_execution: bool = False
    agent_type: str = "semantic_md"
    requires_rigor_refinement: bool = False
    document_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationContext:
    """Unified context for all compilation operations."""
    
    # Input parameters
    source_path: str
    is_folder: bool = False
    run_id: str = ""
    parent_run_id: Optional[str] = None
    domain: str = "SUPPLY_CHAIN"
    skip_rigor: bool = False
    force_llm_extraction: bool = False
    fast_mode: bool = False
    max_files: Optional[int] = None
    output_dir: Optional[str] = None
    analysis_question: Optional[str] = None
    raw_text: str = ""
    
    # [P2-1] 可选：使用 AgentExecutionService 作为备用提取器
    use_agent_execution: bool = False  # 默认关闭
    agent_type: str = "semantic_md"  # semantic_md, data_analysis
    
    # Quality feedback loop: BRONZE triggers RIGOR refinement
    requires_rigor_refinement: bool = False
    
    # Timing tracking
    step_timings: Dict[str, float] = field(default_factory=dict)
    
    # Results
    fragments: List[Fragment] = field(default_factory=list)
    final_fragment: Optional[Fragment] = None
    quality_level: str = "SILVER"
    quality_score: float = 0.0
    quality_fallback_attempted: bool = False
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    cross_file_associations: Dict[str, Any] = field(default_factory=dict)
    verifier_stats: Dict[str, Any] = field(default_factory=dict)
    skill_analysis: Dict[str, Any] = field(default_factory=dict)
    skill_proposals: Dict[str, Any] = field(default_factory=dict)
    skill_reviews: Dict[str, Any] = field(default_factory=dict)
    skill_version_info: Dict[str, Any] = field(default_factory=dict)
    deduplication_stats: Dict[str, int] = field(default_factory=dict)
    change_report: Dict[str, Any] = field(default_factory=dict)
    llm_usage: Dict[str, Any] = field(default_factory=dict)
    
    # Output artifacts
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class PipelineStep(Protocol):
    """Protocol for pipeline steps."""
    
    async def execute(self, ctx: CompilationContext) -> CompilationContext:
        """Execute the step and return updated context."""
        ...


class UnifiedCompilationService:
    """Backward-compatible facade for unified compilation pipelines."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    async def compile_source(self, request: CompilationRequest):
        document_metadata = getattr(request, "document_metadata", {}) or {}
        chunk_strategy = (document_metadata.get("chunk_strategy") or "").strip() or None
        extractor = TrueUnifiedExtractor(chunk_strategy=chunk_strategy)
        if Path(request.source_path).exists():
            suffix = Path(request.source_path).suffix.lower()
            if suffix in [".docx", ".doc", ".pptx", ".ppt", ".pdf", ".xlsx", ".xls", ".csv"]:
                content = FileParser.extract_text(request.source_path)
            else:
                content = Path(request.source_path).read_text(encoding="utf-8", errors="replace")
        else:
            content = request.source_path
        try:
            result = await extractor.extract_async_content(
                content=content,
                domain=request.domain,
                source_file=request.source_path,
                fast_mode=request.fast_mode,
                document_metadata=getattr(request, "document_metadata", {}) or {},
            )
            nodes = result.get("nodes", []) or []
            instances = result.get("instances", []) or []
            relationships = result.get("relationships", []) or []
            metadata = result.get("metadata", {}) or {}
            metadata.setdefault("document_type", getattr(request, "document_metadata", {}).get("document_type", ""))
            metadata.setdefault("document_type_confidence", getattr(request, "document_metadata", {}).get("document_type_confidence", 0.0))
            metadata.setdefault("matched_skills", getattr(request, "document_metadata", {}).get("matched_skills", []))
            metadata.setdefault("skill_stats", getattr(request, "document_metadata", {}).get("skill_stats", {}))
            raw_status = _normalize_status(result.get("status", "SUCCESS"))
            degraded = bool(metadata.get("degraded")) or result.get("llm_usage", {}) == {} or raw_status == "DEGRADED_SUCCESS"
            chunk_quality = str(metadata.get("chunk_quality", "")).upper()
            run_id = request.run_id or Path(request.source_path).stem
            if raw_status in {"RETRYABLE_FAILURE", "CONTRACT_FAILURE", "FATAL_FAILURE", "FAILED"}:
                return CompilationRunResult(
                    success=False,
                    run_id=run_id,
                    quality_level="FAILED",
                    quality_score=float(metadata.get("quality_score", 0.0) or 0.0),
                    nodes=nodes,
                    instances=instances,
                    relationships=relationships,
                    node_count=len(nodes),
                    edge_count=len(relationships),
                    warnings=["offline fallback: extractor returned failure status"],
                    errors=_extract_error_messages(metadata, f"extractor status={raw_status}") + [f"extractor status={raw_status}"],
                    status=raw_status,
                    publishable=False,
                )
            if degraded or (not _has_meaningful_payload(result)):
                quality_level = "DEGRADED"
                if chunk_quality == "PARTIAL_SUCCESS":
                    quality_level = "PARTIAL"
                elif chunk_quality == "DEGRADED_SUCCESS":
                    quality_level = "DEGRADED"
                return CompilationRunResult(
                    success=True,
                    run_id=run_id,
                    quality_level=quality_level,
                    quality_score=float(metadata.get("quality_score", 0.0) or 0.0),
                    nodes=nodes,
                    instances=instances,
                    relationships=relationships,
                    node_count=len(nodes),
                    edge_count=len(relationships),
                    warnings=[degraded_result_warning(result, metadata)],
                    errors=_extract_error_messages(metadata, "degraded result"),
                    status="DEGRADED_SUCCESS" if quality_level == "DEGRADED" else "PARTIAL_SUCCESS",
                    publishable=False,
                )
            return CompilationRunResult(
                success=True,
                run_id=run_id,
                quality_level=metadata.get("quality_level", "SILVER"),
                quality_score=float(metadata.get("quality_score", 0.0) or 0.0),
                nodes=nodes,
                instances=instances,
                relationships=relationships,
                node_count=len(nodes),
                edge_count=len(relationships),
                warnings=[],
                errors=[],
                status="SUCCESS",
                publishable=True,
            )
        except Exception as exc:
            run_id = request.run_id or Path(request.source_path).stem
            return CompilationRunResult(
                success=False,
                run_id=run_id,
                quality_level="FAILED",
                quality_score=0.0,
                nodes=[],
                instances=[],
                relationships=[],
                node_count=0,
                edge_count=0,
                warnings=[],
                errors=[f"offline fallback failed: {exc}"],
                status="FATAL_FAILURE",
                publishable=False,
            )
