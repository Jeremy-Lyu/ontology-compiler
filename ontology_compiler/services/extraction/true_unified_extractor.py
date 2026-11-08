"""
True Unified Extractor

Performs a single LLM call to extract L1, L2, and L2.5 layers simultaneously.
This is the core component for achieving 7-10x speed improvement while
maintaining or improving extraction quality.

**P2-2 Simplification**: Now directly implements ExtractorPlugin interface,
eliminating the need for TrueUnifiedExtractorPlugin wrapper.

Key features:
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import asyncio
from dataclasses import dataclass

from ...agents.extractor_plugin import (
    ExtractorPlugin,
    ExtractorMetadata,
    ExtractorResult,
)
from ontology_compiler.services.ontology_context_loader import OntologyContextLoader
from ontology_compiler.services.extraction.llm_response_parser import parse_unified_response
from ontology_compiler.services.extraction.entity_normalizer import normalize_entities as normalize_entities_layers
from ontology_compiler.services.extraction.extraction_postprocessing import (
    enforce_l3_parents,
)
from ontology_compiler.services.cross_file_association_service import CrossFileAssociationService
from ontology_compiler.services.extraction.extraction_steps import (
    apply_hard_fixes,
    apply_layer_quality_filter,
    apply_parent_id_fixes,
    autogenerate_parent_classes,
    autogenerate_parent_ids,
    build_entity_maps,
    build_name_to_id_map,
    decouple_l1_to_l2_relationships,
    deduplicate_instances,
    deduplicate_nodes,
    enhance_entity,
    normalize_and_filter_relationships,
    reclassify_l1_to_l2,
    reclassify_l2_to_l1,
    validate_l0_entities,
)
from ontology_compiler.services.merge.entity_dedup_service import EntityDeduplicationService
from ontology_compiler.services.quality_assurance.quality_rules import calculate_unified_quality_score
from ontology_compiler.services.l25_skill_loader import L25SkillLoader, get_l25_skill_loader
from ontology_compiler.services.extraction.prompt_loader import get_prompt_loader
from ontology_compiler.services.extraction.llm_call import call_llm
from ontology_compiler.services.validation_service import (
    add_governance_feedback,
    add_validation_feedback,
    infer_entity_class,
    validate_extraction,
    validate_response_semantics,
    validate_with_schema_registry,
)
from ontology_compiler.services.extraction.prompt_service import (
    DocumentAnalyzer,
    PromptService,
    PromptStrategy,
)
from ontology_compiler.services.extraction.chunking_service import extract_with_chunking
from ontology_compiler.services.governance_service import apply_skill_governance
from ontology_compiler.services.relationship_service import (
    build_entity_context_text,
    extract_relationships_separate,
)
from ontology_compiler.services.extraction.refinement_service import refine_l2
from ontology_compiler.services.plugin_adapter import (
    build_failure_result,
    build_sync_extractor_result,
    read_source_content,
)
from ontology_compiler.services.extraction.extraction_failure_memory import (
    apply_failure_memory_adjustments, record_extraction_failure,
    record_schema_violation,  # noqa: F401  (kept for star-import compat)
)
from ontology_compiler.services.extraction.extraction_packet_adapter import maybe_attach_ontology_packet
from ontology_compiler.services.execution_tracer import ExecutionTracer, get_tracer
from ontology_compiler.services.extraction.evidence_locator import EvidenceLocator, enrich_entities_with_evidence
from ontology_compiler.services.extraction.relationship_inferencer import RelationshipInferencer
from ...config import get_config

try:
    from core.schema_registry import get_schema_registry
except ImportError:
    get_schema_registry = None
    logger = logging.getLogger(__name__)
    logger.warning("SchemaRegistry import failed - schema validation will be disabled")

try:
    from core.events import FragmentExtractedEvent, ExtractionFailedEvent
    from core.events.event_consumer_validator import EventConsumerRegistry
    EVENT_ENABLED = True
except ImportError:
    EVENT_ENABLED = False
    logger = logging.getLogger(__name__)
    logger.warning("Domain events import failed - event publishing will be disabled")

logger = logging.getLogger(__name__)

try:
    from ontology_compiler.services.extraction.text_chunking import TextChunker
    TEXT_CHUNKING_ENABLED = True
except ImportError:
    TextChunker = None
    TEXT_CHUNKING_ENABLED = False
    logger.warning("TextChunker import failed - smart chunking will be disabled")


# ---------------------------------------------------------------------------
# C-2: optional PromptBudgetPolicy integration helpers.
#
# ontology_compiler must not import src.application at module load time
# (dependency direction). These helpers lazily import the policy only when
# actually needed, so the module stays importable without src.
# ---------------------------------------------------------------------------


def _try_get_prompt_budget_policy():
    """Return the process-wide PromptBudgetPolicy, or None if unavailable."""
    try:
        from src.application.services.prompt_budget_policy import get_prompt_budget_policy

        return get_prompt_budget_policy()
    except Exception:
        return None


# C-2: chunk-size derivation lives in text_chunking._resolve_chunk_size_from_budget
# so the same helper is shared by TextChunker and the extractor.


# ---------------------------------------------------------------------------
# OCR garbage preflight — cheap heuristic to skip LLM calls for scanned /
# corrupted PDFs whose text layer is mostly noise (unprintable chars, CJK
# fragments, control chars, mojibake). Saves a full LLM round-trip per file.
# ---------------------------------------------------------------------------

# Tunables (can be overridden via env vars):
#   OCR_MIN_SAMPLE_CHARS       — skip check entirely for very short texts
#   OCR_ALPHA_RATIO_THRESHOLD  — printable ASCII letters below this ratio → garbage
#   OCR_PRINTABLE_RATIO_MIN    — printable chars below this → garbage
#   OCR_CONTROL_CHAR_MAX_RATIO — control chars above this → garbage
_OCR_MIN_SAMPLE_CHARS = int(os.environ.get("OCR_MIN_SAMPLE_CHARS", "200"))
_OCR_ALPHA_RATIO_THRESHOLD = float(os.environ.get("OCR_ALPHA_RATIO_THRESHOLD", "0.50"))
_OCR_PRINTABLE_RATIO_MIN = float(os.environ.get("OCR_PRINTABLE_RATIO_MIN", "0.85"))
_OCR_CONTROL_CHAR_MAX_RATIO = float(os.environ.get("OCR_CONTROL_CHAR_MAX_RATIO", "0.02"))


def is_ocr_garbage(text: str) -> Tuple[bool, str]:
    """Return (is_garbage, reason_string) for a text sample.

    A 4-signal cheap classifier. Intentionally conservative: we only flag if
    multiple signals agree the text is unusable. False negatives (sending
    garbage to LLM) waste money/time but don't crash the pipeline; false
    positives (skipping valid text) would silently lose data. So we AND the
    two strongest signals: alpha_ratio < threshold **AND** printable_ratio
    low. Control-char ratio is a secondary tiebreaker.
    """
    if not text:
        return True, "empty_text"
    sample_len = len(text)
    if sample_len < _OCR_MIN_SAMPLE_CHARS:
        # Too short to reliably classify; let the LLM handle it.
        return False, f"too_short(sample={sample_len}<{_OCR_MIN_SAMPLE_CHARS})"

    # Signal 1: printable ASCII letter ratio (a-z A-Z). Good English docs
    # are ~60-70 %; Chinese mixed docs are ~20-40 %; OCR garbage is ~0-15 %.
    letters = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    alpha_ratio = letters / sample_len

    # Signal 2: printable ratio. OCR junk is full of control chars and
    # unprintable bytes.
    printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
    printable_ratio = printable / sample_len

    # Signal 3: control-char ratio (excluding the common whitespace ones).
    control = sum(
        1 for c in text
        if (ord(c) < 32 and c not in "\r\n\t") or ord(c) == 127
    )
    control_ratio = control / sample_len

    # Decision: require BOTH alpha AND printable to fail (avoids false
    # positives on pure CJK docs which legitimately have low alpha_ratio).
    reasons: list[str] = []
    if alpha_ratio < _OCR_ALPHA_RATIO_THRESHOLD:
        reasons.append(f"alpha_ratio={alpha_ratio:.2f}<{_OCR_ALPHA_RATIO_THRESHOLD}")
    if printable_ratio < _OCR_PRINTABLE_RATIO_MIN:
        reasons.append(f"printable_ratio={printable_ratio:.2f}<{_OCR_PRINTABLE_RATIO_MIN}")
    if control_ratio > _OCR_CONTROL_CHAR_MAX_RATIO:
        reasons.append(f"control_ratio={control_ratio:.3f}>{_OCR_CONTROL_CHAR_MAX_RATIO}")

    # Garbage if: (alpha low AND printable low) OR (control chars absurdly high).
    garbage = (
        (alpha_ratio < _OCR_ALPHA_RATIO_THRESHOLD and printable_ratio < _OCR_PRINTABLE_RATIO_MIN)
        or (control_ratio > _OCR_CONTROL_CHAR_MAX_RATIO)
    )
    reason = "; ".join(reasons) if reasons else f"ok(alpha={alpha_ratio:.2f},printable={printable_ratio:.2f},ctrl={control_ratio:.3f})"
    return garbage, reason


@dataclass
class UnifiedExtractionResult:
    """Result from unified extraction containing all five layers + relationships."""
    l0_entities: List[Dict[str, Any]]
    l1_entities: List[Dict[str, Any]]
    l2_entities: List[Dict[str, Any]]
    l25_entities: List[Dict[str, Any]]
    l3_entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    raw_response: str
    processing_time_ms: float
    token_usage: Dict[str, int]
    quality_score: float = 0.0
    degraded: bool = False  # Phase 1: Flag for degraded mode results




class TrueUnifiedExtractor(ExtractorPlugin):
    """
    True Unified Extractor - extracts L1, L2, and L2.5 in a single LLM call.
    
    **P2-2 Simplification**: Directly implements ExtractorPlugin interface.
    **P2-4 Integration**: Now includes SmartRefiner L2 refinement capability.
    
    This extractor achieves speed improvement by:
    1. Single LLM call instead of 3-4 separate calls
    2. Optimized prompt design for comprehensive extraction
    3. Async processing support for concurrent execution
    4. L2.5 Skill integration for domain-specific extraction rules
    5. Built-in L2 refinement for enhanced business scenario extraction
    
    Expected performance:
    - Previous: ~60-80 seconds (3-4 LLM calls)
    - New: ~15-25 seconds (1 LLM call)
    - Speedup: ~3-4x from call reduction alone
    """
    
    def __init__(
        self,
        llm_client = None,
        context_loader: Optional[OntologyContextLoader] = None,
        skill_loader: Optional[L25SkillLoader] = None,
        max_tokens: int = 32000,
        temperature: float = 0.2,
        timeout: int = 600,  # [FIX] 与流式API匹配：流式read timeout=300s(per-chunk)防卡死，外层600s允许32K tokens完整生成
        config: Optional[Dict[str, Any]] = None,
        enable_refinement: bool = False,
        refinement_threshold: int = 3,
        tracer: Optional[ExecutionTracer] = None,
        chunk_strategy: Optional[str] = None,
        budget_policy = None,
    ):
        """
        Initialize the True Unified Extractor.

        Auto-selects LLM client based on config.llm_provider (deepseek or dify).

        Args:
            llm_client: Optional pre-configured LLM client instance
            context_loader: OntologyContextLoader for master ontology reference
            skill_loader: L25SkillLoader for domain-specific extraction rules (optional)
            max_tokens: Maximum tokens for LLM response (default 8000 for full extraction)
            temperature: LLM temperature (lower for consistent structured output)
            config: Optional configuration dict (ExtractorPlugin interface)
            enable_refinement: Enable L2 refinement (default False)
            refinement_threshold: Minimum entities before refinement triggers (default 3)
            tracer: Optional ExecutionTracer for end-to-end tracing
            chunk_strategy: Optional explicit chunking strategy override
                ("slide_based" / "section_based" / "hybrid"); auto-selected when
                not provided.
            budget_policy: Optional PromptBudgetPolicy (C-2) for task-aware
                token/temperature defaults. When provided, max_tokens and
                temperature defaults are sourced from the extraction_technical
                profile unless overridden via config or kwargs.
        """
        super().__init__(config or {})
        
        if llm_client:
            self.llm = llm_client
        else:
            # D-1 (2026-08-15): use the shared client factory so circuit-
            # breaker / rate-limiter / cache state is shared across callers.
            # Factory lives in ontology_compiler.rigor (dependency direction).
            from ...rigor.llm_backend import get_llm_client
            self.llm = get_llm_client()
            fallback = getattr(self.llm, "_fallback_clients", [])
            fallback_names = [name for name, _ in fallback]
            logger.info(
                "TrueUnifiedExtractor using UnifiedLLMClient via factory (primary=%s, fallbacks=%s)",
                getattr(self.llm, "provider", "unknown"),
                fallback_names,
            )
        
        # C-2: optional PromptBudgetPolicy for task-aware token / temperature /
        # retry defaults. Falls back to environment-driven default when unset.
        self._budget_policy = budget_policy or _try_get_prompt_budget_policy()

        # Allow config override for max_tokens and temperature. When a config
        # override is absent and a budget policy is attached, use the policy's
        # extraction_technical profile so the per-call cap matches the YAML
        # budget rather than the legacy 32K default.
        cfg_has_max_tokens = bool(config) and "max_tokens" in (config or {})
        cfg_has_temperature = bool(config) and "temperature" in (config or {})
        if cfg_has_max_tokens:
            self.max_tokens = self.config.get("max_tokens", max_tokens)
        elif self._budget_policy is not None:
            _budget = self._budget_policy.resolve(task="extraction_technical")
            self.max_tokens = _budget.max_completion_tokens
        else:
            self.max_tokens = max_tokens
        if cfg_has_temperature:
            self.temperature = self.config.get("temperature", temperature)
        elif self._budget_policy is not None:
            _budget = self._budget_policy.resolve(task="extraction_technical")
            self.temperature = _budget.temperature
        else:
            self.temperature = temperature
        self.timeout = self.config.get("timeout", timeout)
        
        self.context_loader = context_loader or OntologyContextLoader()
        self.skill_loader = skill_loader or get_l25_skill_loader()
        self.deduplication_service = EntityDeduplicationService()
        self.association_service = CrossFileAssociationService()
        self.master_context = None
        
        # P2-4: Refinement configuration
        self.enable_refinement = self.config.get("enable_refinement", enable_refinement)
        self.refinement_threshold = self.config.get("refinement_threshold", refinement_threshold)
        
        # [PIPELINE-IMPROVEMENT]: Dynamic smart prompt design
        self.document_analyzer = DocumentAnalyzer()
        self.prompt_strategy = PromptStrategy()
        self.prompt_loader = get_prompt_loader()
        self._prompt_service = PromptService(
            document_analyzer=self.document_analyzer,
            context_loader=self.context_loader,
            prompt_strategy=self.prompt_strategy,
            prompt_loader=self.prompt_loader,
            skill_loader=self.skill_loader,
            budget_policy=self._budget_policy,
        )
        
        # [OPTIMIZATION]: Smart text chunking for large documents
        self.chunker = TextChunker(budget_policy=self._budget_policy) if TEXT_CHUNKING_ENABLED else None
        self.chunk_strategy = (chunk_strategy or "").strip() or None
        
        # [Phase 6-ARCH-2] End-to-end tracing
        self.tracer = tracer or get_tracer()
        self._current_run_id: Optional[str] = None
        
        # Statistics tracking
        self._call_count = 0
        self._total_tokens = 0
        self._total_time = 0.0
    
    def get_metadata(self) -> ExtractorMetadata:
        """Return metadata about this extractor (ExtractorPlugin interface)."""
        return ExtractorMetadata(
            name="true_unified_extractor",
            version="2.1.0",
            description="Unified L1/L2/L2.5 extractor in single LLM call (P2-2 simplified)",
            author="Ontology Team",
            supported_formats=[".md", ".txt", ".pdf", ".docx"],
            capabilities=["L1_extraction", "L2_extraction", "L2.5_extraction", "unified_call"],
            tags=["unified", "llm_based", "all_layers"]
        )
    
    def can_extract(self, source: str | Path, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if this extractor can handle the given source (ExtractorPlugin interface)."""
        source_str = str(source)
        supported_extensions = (".md", ".txt", ".pdf", ".docx", ".doc")
        return any(source_str.lower().endswith(ext) for ext in supported_extensions)
    
    async def _refine_l2(self, content: str, initial_l2: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """P2-4: Refine L2 entities by making additional LLM call for enhanced extraction.

        Extracted to ``refinement_service.refine_l2`` (2026-08-11).
        """
        return await refine_l2(
            content, initial_l2, domain,
            llm=self.llm,
            refinement_threshold=self.refinement_threshold,
            enable_refinement=self.enable_refinement,
        )

    async def extract_async_content(
        self,
        content: str,
        domain: Optional[str] = None,
        source_file: Optional[str] = None,
        fast_mode: bool = False,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Async pipeline wrapper: accepts pre-extracted text content directly.

        NOTE: Named extract_async_content (not extract_async) to avoid being
        shadowed by the ExtractorPlugin-compatible extract_async(source, context)
        defined further below. Called by TrueUnifiedExtractStep._extract_fragment().

        Args:
            content: Document text to extract from
            domain: Domain hint (e.g., "SUPPLY_CHAIN")
            source_file: Source file path for context
            fast_mode: Skip LLM call, return empty result

        Returns:
            Dict with nodes, relationships, instances, metadata, and llm_usage
        """
        try:
            file_path = Path(source_file) if source_file else None
            result = await self.extract_all_layers(
                text=content,
                file_path=file_path,
                domain=domain,
                use_context=False,
                enable_validation=True,
                max_retries=2,
                fast_mode=fast_mode,
                document_metadata=document_metadata or {}
            )
            
            print(f"[DEBUG] extract_all_layers returned: L0={len(result.l0_entities)}, L1={len(result.l1_entities)}, L2={len(result.l2_entities)}, L2.5={len(result.l25_entities)}, L3={len(result.l3_entities)}, rels={len(result.relationships)}")

            source_file_str = str(file_path) if file_path else ""

            try:
                from core.schema_registry import get_schema_registry
                registry = get_schema_registry()
                VALID_L0_CLASSES = set(registry.main_classes)
            except ImportError:
                VALID_L0_CLASSES = {"Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"}

            # Stage 1: enhance per-layer entities (tag layer/source_file,
            # normalize type + evidence) and validate L0 membership.
            all_entities = validate_l0_entities(result.l0_entities, VALID_L0_CLASSES, source_file_str)
            for entities, layer in (
                (result.l1_entities, "L1"),
                (result.l2_entities, "L2"),
                (result.l25_entities, "L2.5"),
            ):
                all_entities.extend(enhance_entity(e, layer, source_file_str) for e in entities)
            instances = [enhance_entity(e, "L3", source_file_str) for e in result.l3_entities]

            # Stage 2: dedup entities/instances with sequential prefixed IDs.
            counter = {"L0": 1, "L1": 1, "L2": 1, "L2.5": 1, "L3": 1}
            id_mapping = {}
            seen_ids = {}
            deduplicated_nodes = deduplicate_nodes(all_entities, counter, id_mapping, seen_ids)
            deduplicated_instances = deduplicate_instances(instances, counter, id_mapping, seen_ids)

            # Stage 3: layer re-classification + quality filter.
            # P1-01: L0 auto-supplement moved to extract.py (added once at the
            # extract stage after all files are processed).
            entity_layer_map, entity_map, l0_class_to_id = build_entity_maps(
                deduplicated_nodes, deduplicated_instances)
            reclassify_l1_to_l2(deduplicated_nodes, seen_ids, counter, id_mapping, l0_class_to_id, VALID_L0_CLASSES)
            reclassify_l2_to_l1(deduplicated_nodes, counter, id_mapping)
            deduplicated_nodes = apply_layer_quality_filter(deduplicated_nodes, counter, id_mapping, result.relationships)
            valid_node_ids = {node["meta_id"] for node in deduplicated_nodes + deduplicated_instances}

            # Stage 4: parent_id auto-generation + dangling reference repair.
            autogenerate_parent_ids(deduplicated_nodes, l0_class_to_id)
            autogenerate_parent_classes(deduplicated_nodes)
            name_to_id_map = build_name_to_id_map(deduplicated_nodes, deduplicated_instances)
            apply_parent_id_fixes(deduplicated_nodes, deduplicated_instances, valid_node_ids,
                                  id_mapping, name_to_id_map, deduplicated_instances)

            # Stage 5: relationship normalization, L1->L2 decoupling, hard fixes.
            enforce_l3_parents(deduplicated_nodes, result.relationships)
            normalized_relationships = normalize_and_filter_relationships(
                result, deduplicated_nodes, deduplicated_instances, valid_node_ids,
                id_mapping, name_to_id_map, entity_map, entity_layer_map)
            decouple_l1_to_l2_relationships(normalized_relationships, deduplicated_nodes,
                                            entity_layer_map, valid_node_ids, entity_map)
            apply_hard_fixes(deduplicated_nodes, deduplicated_instances, normalized_relationships,
                            entity_map, entity_layer_map, valid_node_ids, id_mapping)

            print(f"[DEBUG] extract_async_content returning: nodes={len(deduplicated_nodes)}, instances={len(deduplicated_instances)}, relationships={len(normalized_relationships)}")
            result_metadata = getattr(result, "metadata", {}) or {}
            return {
                'nodes': deduplicated_nodes,
                'relationships': normalized_relationships,
                'instances': deduplicated_instances,
                'metadata': {
                    'quality_score': result.quality_score,
                    'processing_time_ms': result.processing_time_ms,
                    'degraded': result.degraded,
                    'source_file': source_file,
                    'extraction_quality_status': result_metadata.get('extraction_quality_status'),
                    'validation_passed': result_metadata.get('validation_passed'),
                    'validation_reason': result_metadata.get('validation_reason'),
                    'chunk_quality': result_metadata.get('chunk_quality'),
                    'chunk_metrics': result_metadata.get('chunk_metrics'),
                },
                'llm_usage': result.token_usage
            }
        except Exception as e:
            logger.error(f"Extraction failed in extract_async_content() wrapper: {e}")
            return {
                'nodes': [],
                'relationships': [],
                'instances': [],
                'metadata': {'error': str(e), 'source_file': source_file},
                'llm_usage': {}
            }
    
    def extract(
        self,
        source: str | Path,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractorResult:
        """
        Extract ontology using unified extraction (ExtractorPlugin interface).
        
        Args:
            source: Source file path or content string
            context: Optional context with keys:
                - domain: str (e.g., "SWDL", "BOM")
                - use_context: bool (default True)
                - file_path: str (for evidence binding)
                
        Returns:
            ExtractorResult with L1, L2, and L2.5 entities
        """
        start_time = __import__('time').time()
        
        ctx = context or {}
        domain = ctx.get("domain")
        use_context = ctx.get("use_context", True)
        file_path = ctx.get("file_path") or (source if isinstance(source, (str, Path)) else None)
        fast_mode = ctx.get("fast_mode", False)
        
        content, read_error = read_source_content(source, start_time, self.get_metadata().name)
        if read_error is not None:
            return read_error
        
        # [P3-EXP-3]: Check failure memory for strategy adjustment recommendations
        # (implementation moved to extraction_failure_memory module, Phase 3
        # LOC reduction — docs/CODE_CLEANUP_CANDIDATES.md §3.1).
        document_key = str(file_path) if file_path else str(content)[:100]
        fast_mode = apply_failure_memory_adjustments(document_key, ctx)["fast_mode"]
        
        # Run async extraction synchronously
        try:
            result = asyncio.run(self.extract_all_layers(
                text=str(content),
                file_path=Path(file_path) if file_path else None,
                domain=domain,
                use_context=use_context,
                fast_mode=fast_mode
            ))
            
            # Convert to ExtractorResult format
            execution_time = (__import__('time').time() - start_time) * 1000
            
            return build_sync_extractor_result(result, domain, file_path, ctx, self.get_metadata().name)
            
        except Exception as e:
            execution_time = (__import__('time').time() - start_time) * 1000
            error_msg = f"Extraction failed: {e}"
            logger.error(error_msg)
            
            # Publish ExtractionFailedEvent
            if EVENT_ENABLED:
                try:
                    event = ExtractionFailedEvent(
                        source_file=str(file_path) if file_path else "memory",
                        error=error_msg,
                        extractor_name=self.get_metadata().name,
                        run_id=ctx.get("run_id"),
                    )
                    logger.info(f"Published ExtractionFailedEvent: {event}")
                except Exception as event_err:
                    logger.warning(f"Failed to publish ExtractionFailedEvent: {event_err}")
            
            record_extraction_failure(
                error_message=error_msg,
                document_type=str(file_path).split('.')[-1] if file_path else "unknown",
                skill_used=self.get_metadata().name,
                layer="L1/L2/L2.5",
                document_key=str(file_path) if file_path else None,
            )
            
            return build_failure_result(error_msg, execution_time, self.get_metadata().name)

    async def extract_async(
        self,
        source: str | Path,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractorResult:
        """Async version of extract() - use when already in an async context."""
        start_time = __import__('time').time()
        
        ctx = context or {}
        domain = ctx.get("domain")
        use_context = ctx.get("use_context", True)
        file_path = ctx.get("file_path") or (source if isinstance(source, (str, Path)) else None)
        fast_mode = ctx.get("fast_mode", False)
        
        content, read_error = read_source_content(source, start_time, self.get_metadata().name)
        if read_error is not None:
            return read_error
        
        try:
            result = await self.extract_all_layers(
                text=str(content),
                file_path=Path(file_path) if file_path else None,
                domain=domain,
                use_context=use_context,
                fast_mode=fast_mode
            )
            
            from src.domain.models import Fragment, FragmentNode, FragmentInstance, FragmentRelationship
            
            nodes = []
            for entity in result.l0_entities:
                nodes.append(FragmentNode(
                    meta_id=entity.get("meta_id", ""),
                    meta_name=entity.get("name", ""),
                    meta_class=entity.get("meta_class", "Capability"),
                    domain=domain or "SUPPLY_CHAIN",
                    source_file=str(file_path) if file_path else "",
                    description=entity.get("description", ""),
                    layer="L0",
                    properties={k: v for k, v in entity.items() if k not in ("meta_id", "name", "meta_class", "description")}
                ))
            
            for entity in result.l1_entities:
                nodes.append(FragmentNode(
                    meta_id=entity.get("meta_id", ""),
                    meta_name=entity.get("name", ""),
                    meta_class=entity.get("meta_class", "System"),
                    domain=domain or "SUPPLY_CHAIN",
                    source_file=str(file_path) if file_path else "",
                    description=entity.get("description", ""),
                    layer="L1",
                    properties={k: v for k, v in entity.items() if k not in ("meta_id", "name", "meta_class", "description")}
                ))
            
            for entity in result.l2_entities:
                nodes.append(FragmentNode(
                    meta_id=entity.get("meta_id", ""),
                    meta_name=entity.get("name", entity.get("scenario_name", "")),
                    meta_class=entity.get("meta_class", "System"),
                    domain=domain or "SUPPLY_CHAIN",
                    source_file=str(file_path) if file_path else "",
                    description=entity.get("description", ""),
                    layer="L2",
                    properties={k: v for k, v in entity.items() if k not in ("meta_id", "name", "scenario_name", "meta_class", "description")}
                ))
            
            for entity in result.l3_entities:
                instances.append(FragmentInstance(
                    meta_id=entity.get("meta_id", ""),
                    meta_name=entity.get("name", ""),
                    meta_class=entity.get("meta_class", "Asset") if entity.get("meta_class") not in ("Instance", "") else "Asset",
                    domain=domain or "SUPPLY_CHAIN",
                    source_file=str(file_path) if file_path else "",
                    description=entity.get("description", ""),
                    layer="L3",
                    properties={k: v for k, v in entity.items() if k not in ("meta_id", "name", "meta_class", "description")}
                ))
            
            for entity in result.l25_entities:
                nodes.append(FragmentNode(
                    meta_id=entity.get("meta_id", ""),
                    meta_name=entity.get("name", ""),
                    meta_class=entity.get("meta_class", "Data"),
                    domain=domain or "SUPPLY_CHAIN",
                    source_file=str(file_path) if file_path else "",
                    description=entity.get("description", ""),
                    layer="L2.5",
                    properties={k: v for k, v in entity.items() if k not in ("meta_id", "name", "meta_class", "description")}
                ))
            
            # Build relationships from LLM-extracted relationships (not l25_entities)
            from core.utils.relationship_helpers import (
                collect_entity_ids,
                is_valid_relationship,
                relationship_endpoints,
            )

            entity_ids = collect_entity_ids(nodes + instances)
            relationships = []
            for rel in result.relationships:
                if isinstance(rel, dict):
                    source_id, target_id = relationship_endpoints(rel)
                    if not is_valid_relationship(rel, entity_ids):
                        continue
                    
                    confidence_val = rel.get("confidence", "MEDIUM")
                    if isinstance(confidence_val, str):
                        confidence_val = {
                            "HIGH": 0.9,
                            "MEDIUM": 0.7,
                            "LOW": 0.4
                        }.get(confidence_val.upper(), 0.7)
                    
                    relationships.append(FragmentRelationship(
                        source=source_id,
                        target=target_id,
                        type=rel.get("type", rel.get("rel_type", "related_to")),
                        domain=domain or "SUPPLY_CHAIN",
                        source_file=str(file_path) if file_path else "",
                        evidence_ref=rel.get("evidence_ref", ""),
                        citation=rel.get("citation", ""),
                        rule_no=rel.get("rule_no", ""),
                        confidence=confidence_val,
                        properties={k: v for k, v in rel.items() if k not in ("from", "to", "source", "target", "type", "rel_type", "evidence_ref", "citation", "rule_no", "confidence")}
                    ))
                else:
                    source_id = getattr(rel, "source", getattr(rel, "from", ""))
                    target_id = getattr(rel, "target", getattr(rel, "to", ""))
                    if source_id not in entity_ids or target_id not in entity_ids or source_id == target_id:
                        continue
                    relationships.append(FragmentRelationship(
                        source=source_id,
                        target=target_id,
                        type=getattr(rel, "type", getattr(rel, "rel_type", "related_to")),
                        domain=domain or "SUPPLY_CHAIN",
                        source_file=str(file_path) if file_path else "",
                        evidence_ref=getattr(rel, "evidence_ref", ""),
                        citation=getattr(rel, "citation", ""),
                        rule_no=getattr(rel, "rule_no", "")
                    ))
            
            # Enrich relationships using RelationshipInferencer
            all_entities = (result.l0_entities or []) + (result.l1_entities or []) + (result.l2_entities or []) + (result.l25_entities or []) + (result.l3_entities or [])
            if len(all_entities) >= 2:
                inferencer = RelationshipInferencer()
                inferred_rels = inferencer.infer_relationships(all_entities, result.relationships)
                
                for rel in inferred_rels:
                    source_id = rel.get("from", "")
                    target_id = rel.get("to", "")
                    if source_id in entity_ids and target_id in entity_ids and source_id != target_id:
                        rel_type = rel.get("type", "related_to")
                        exists = any(r.source == source_id and r.target == target_id and r.type == rel_type for r in relationships)
                        if not exists:
                            relationships.append(FragmentRelationship(
                                source=source_id,
                                target=target_id,
                                type=rel_type,
                                domain=domain or "SUPPLY_CHAIN",
                                source_file=str(file_path) if file_path else "",
                                evidence_ref=rel.get("evidence_ref", ""),
                                citation=rel.get("citation", ""),
                                rule_no=rel.get("rule_no", ""),
                                properties={"confidence": rel.get("confidence", "MEDIUM")}
                            ))
            
            fragment = Fragment(
                source_file=str(file_path) if file_path else "",
                domain=domain or "SUPPLY_CHAIN",
                nodes=nodes,
                instances=instances,
                relationships=relationships,
                metadata={
                    "l1_count": len(result.l1_entities),
                    "l2_count": len(result.l2_entities),
                    "l25_count": len(result.l25_entities),
                    "quality_score": getattr(result, 'quality_score', 0.0),
                    "token_usage": getattr(result, 'token_usage', {}),
                    "domain": domain,
                    "extractor_name": self.get_metadata().name
                }
            )
            
            execution_time = (__import__('time').time() - start_time) * 1000
            return ExtractorResult(
                success=True,
                fragment=fragment,
                execution_time_ms=execution_time,
                extractor_name=self.get_metadata().name
            )
            
        except Exception as e:
            execution_time = (__import__('time').time() - start_time) * 1000
            error_msg = f"Extraction failed: {e}"
            logger.error(error_msg)
            
            record_extraction_failure(
                error_message=error_msg,
                document_type=str(file_path).split('.')[-1] if file_path else "unknown",
                skill_used=self.get_metadata().name,
                layer="L1/L2/L2.5",
                document_key=str(file_path) if file_path else None,
            )
            
            return build_failure_result(error_msg, execution_time, self.get_metadata().name)

    def _build_entity_context_text(self, text: str, entities_list: list) -> str:
        """Build entity-centric context text for relationship extraction of large documents.

        Extracted to ``relationship_service.build_entity_context_text`` (2026-08-11).
        """
        return build_entity_context_text(text, entities_list)

    async def _extract_relationships_separate(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Two-phase extraction: Extract relationships separately after entities are extracted.

        Extracted to ``relationship_service.extract_relationships_separate`` (2026-08-11).
        """
        return await extract_relationships_separate(
            text, entities, domain,
            llm=self.llm,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
            prompt_loader=self.prompt_loader,
            build_entity_context_text=build_entity_context_text,
        )

    async def extract_unified(
        self,
        text: str,
        file_path: Optional[str] = None,
        domain: Optional[str] = None,
        use_context: bool = True,
        enable_validation: bool = True,
        max_retries: int = 3,
        fast_mode: bool = False,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> UnifiedExtractionResult:
        """
        Extract L1, L2, and L2.5 layers in a single LLM call.
        
        Args:
            text: Document text to extract from
            file_path: Optional file path for context
            domain: Optional domain hint (e.g., "SWDL", "BOM")
            use_context: Whether to include master ontology context
            enable_validation: Enable JSON Schema validation (default: True)
            max_retries: Max retries on validation failure (default: 2)
            fast_mode: Skip LLM call, return empty result for pipeline testing
            
        Returns:
            UnifiedExtractionResult with all three layers
        """
        import time
        start_time = time.time()

        # [OPTIMIZATION 0] OCR garbage preflight — before ANY LLM / chunking /
        # prompt building work, detect texts that are clearly unusable (scanned
        # PDFs with no text layer, corrupted encoding, etc.). This saves a full
        # LLM round-trip plus ~1s of prompt-service work per garbage file.
        _garbage, _garbage_reason = is_ocr_garbage(text)
        if _garbage:
            _fp = file_path or "<unknown>"
            logger.warning(
                "[EXTRACT] OCR garbage detected — skipping LLM call | file=%s "
                "len=%d reason=%s", _fp, len(text), _garbage_reason,
            )
            print(f"[EXTRACT] 🗑️ OCR garbage detected, skipping LLM | file={Path(_fp).name if file_path else 'n/a'} len={len(text)} reason={_garbage_reason}")
            return UnifiedExtractionResult(
                l0_entities=[],
                l1_entities=[],
                l2_entities=[],
                l25_entities=[],
                l3_entities=[],
                relationships=[],
                raw_response=f"__SKIPPED_OCR_GARBAGE__:{_garbage_reason}",
                processing_time_ms=round((time.time() - start_time) * 1000, 1),
                token_usage={},
                quality_score=0.0,
                degraded=True,
            )

        if fast_mode:
            logger.info(f"Fast mode enabled - skipping LLM extraction for {len(text)} chars")
            return UnifiedExtractionResult(
                l0_entities=[],
                l1_entities=[],
                l2_entities=[],
                l25_entities=[],
                l3_entities=[],
                relationships=[],
                raw_response="",
                processing_time_ms=0,
                token_usage={},
                quality_score=0.0,
                degraded=True,
            )

        print(f"[EXTRACT] Starting unified extraction | Text: {len(text)} chars | Domain: {domain} | OCR-qc: {_garbage_reason}")

        # [OPTIMIZATION]: Smart chunking for large documents.
        # CHUNK_THRESHOLD_CHARS env override (default 8000) keeps 90% of
        # EDH/SWDL docs in a single LLM call. Batching 4 files at Semaphore(4)
        # means we want individual chunks to complete in < 3 min so progress
        # is visible; 8000 chars → ~3-5K output tokens fits nicely.
        _CHUNK_THRESHOLD = int(os.environ.get("CHUNK_THRESHOLD_CHARS", "8000") or 8000)
        if self.chunker and len(text) > _CHUNK_THRESHOLD:
            return await self._extract_with_chunking(
                text, file_path, domain, use_context, enable_validation, max_retries,
            )
        
        # Build comprehensive prompt
        prompt_metrics: Dict[str, Any] = {}
        prompt = self._build_unified_prompt(text, domain, use_context, document_metadata=document_metadata, metrics=prompt_metrics)
        print(f"[EXTRACT] Prompt: {len(prompt)} chars | truncation: {prompt_metrics.get('truncation_reason', 'none')} (dropped={prompt_metrics.get('dropped_chars', 0)})")
        
        # Single LLM call with validation and retry
        logger.info(f"Performing unified extraction for {len(text)} chars, domain={domain}")
        
        last_error = None
        (l0_entities, l1_entities, l2_entities, l25_entities, l3_entities, relationships,
         response_text, result_degraded, attempt, text, prompt) = await self._run_llm_extraction_loop(
            prompt=prompt, text=text, domain=domain, use_context=use_context,
            max_retries=max_retries, enable_validation=enable_validation,
            document_metadata=document_metadata,
        )

        
        processing_time = (time.time() - start_time) * 1000
        
        # Post-process against master ontology context when available
        if use_context and self.context_loader:
            master_context = self.context_loader.load_master_ontology()
            master_entities = master_context.all_entities.get("L1", []) + master_context.all_entities.get("L2", []) + master_context.all_entities.get("L2.5", [])
            new_entities = l1_entities + l2_entities + l25_entities
            duplicate_matches = self.deduplication_service.deduplicate_with_master(new_entities, master_entities)
            if duplicate_matches:
                logger.info(f"Matched {len(duplicate_matches)} entities against master ontology")
            same_as_relationships = self.association_service.find_same_as_relationships(new_entities, master_entities)
            if same_as_relationships:
                relationships.extend([
                    {
                        "from": rel.source_id,
                        "to": rel.target_id,
                        "type": rel.relationship_type,
                        "description": "; ".join(rel.evidence),
                    }
                    for rel in same_as_relationships
                ])

        # Optional refinement for insufficient L2 coverage
        if self.enable_refinement:
            l2_entities = await self._quality_refine_if_needed(text, l2_entities, domain)

        # [R7 - EVIDENCE_PRECISION]: Enrich entities with source location evidence
        if text and file_path:
            try:
                l0_enriched = enrich_entities_with_evidence(
                    l0_entities, text, file_path, entity_type="L0"
                )
                l1_enriched = enrich_entities_with_evidence(
                    l1_entities, text, file_path, entity_type="L1"
                )
                l2_enriched = enrich_entities_with_evidence(
                    l2_entities, text, file_path, entity_type="L2"
                )
                l25_enriched = enrich_entities_with_evidence(
                    l25_entities, text, file_path, entity_type="L2.5"
                )
                l3_enriched = enrich_entities_with_evidence(
                    l3_entities, text, file_path, entity_type="L3"
                )
                logger.info(
                    f"Evidence enrichment complete for "
                    f"L0={len(l0_enriched)}/{len(l0_entities)}, L1={len(l1_enriched)}/{len(l1_entities)}, "
                    f"L2={len(l2_enriched)}/{len(l2_entities)}, L2.5={len(l25_enriched)}/{len(l25_entities)}, "
                    f"L3={len(l3_enriched)}/{len(l3_entities)}"
                )
            except Exception as e:
                logger.warning(f"Evidence enrichment failed: {e}")

        # [ENTITY_NORMALIZATION]: Auto-fill missing status fields and normalize naming conventions
        l0_entities, l1_entities, l2_entities, l25_entities, l3_entities = self._normalize_entities(
            l0_entities, l1_entities, l2_entities, l25_entities, l3_entities
        )

        # [PHASE-3 SKILL GOVERNANCE]: Post-extraction governance validation
        skill_governance_report = self._apply_skill_governance(
            text=text,
            domain=domain,
            l1_entities=l1_entities,
            l2_entities=l2_entities,
            l25_entities=l25_entities,
            relationships=relationships,
        )
        if skill_governance_report.get("issues"):
            mode = skill_governance_report.get("mode", "audit_only")
            issues = skill_governance_report["issues"]
            if mode == "strict":
                logger.warning(
                    f"Skill governance (strict): {len(issues)} issue(s) detected — "
                    f"blocked_relation_violations={skill_governance_report.get('blocked_relations', 0)} "
                    f"entity_type_violations={skill_governance_report.get('entity_type_violations', 0)}"
                )
            else:
                logger.info(
                    f"Skill governance ({mode}): {len(issues)} issue(s) "
                    f"(blocked={skill_governance_report.get('blocked_relations', 0)} "
                    f"entity_violations={skill_governance_report.get('entity_type_violations', 0)})"
                )

            # [GOVERNANCE-FEEDBACK-RETRY]: 如果有违规且允许重试，将违规信息注入 prompt 重新提取
            if mode != "audit_only" and skill_governance_report.get("blocked_relations", 0) > 0:
                governance_retry_count = getattr(self, '_governance_retry_count', 0)
                max_governance_retries = 2
                if governance_retry_count < max_governance_retries:
                    logger.info(f"Skill governance violations detected, initiating retry {governance_retry_count + 1}/{max_governance_retries}")
                    prompt = add_governance_feedback(prompt, skill_governance_report)
                    try:
                        llm_result = await self._call_llm(prompt)
                        response_text = llm_result["response"]
                        l0_entities, l1_entities, l2_entities, l25_entities, l3_entities, relationships = self._parse_unified_response(response_text)
                        setattr(self, '_governance_retry_count', governance_retry_count + 1)
                        
                        skill_governance_report = self._apply_skill_governance(
                            text=text,
                            domain=domain,
                            l1_entities=l1_entities,
                            l2_entities=l2_entities,
                            l25_entities=l25_entities,
                            relationships=relationships,
                        )
                        logger.info(f"Governance retry completed, remaining issues: {len(skill_governance_report.get('issues', []))}")
                    except Exception as e:
                        logger.warning(f"Governance retry failed: {e}")

        # [SCHEMA_VALIDATION] Validate meta_class and meta_subclass against SchemaRegistry
        if enable_validation and get_schema_registry:
            print(f"[DEBUG] Before schema validation: L1={len(l1_entities)}, L2={len(l2_entities)}, L2.5={len(l25_entities)}")
            l1_entities, l2_entities, l25_entities, relationships = validate_with_schema_registry(
                l1_entities, l2_entities, l25_entities, relationships, domain
            )
            print(f"[DEBUG] After schema validation: L1={len(l1_entities)}, L2={len(l2_entities)}, L2.5={len(l25_entities)}")

        # [TWO-PHASE EXTRACTION]: Perform separate relation extraction to ensure relationships are captured.
        # Controlled by env var EXTRACT_SKIP_TWO_PHASE=1 to skip for faster runs (saves ~60-120s).
        all_entities = l0_entities + l1_entities + l2_entities + l25_entities + l3_entities
        _skip_two_phase = os.environ.get("EXTRACT_SKIP_TWO_PHASE", "0") == "1"
        if len(all_entities) > 0 and not _skip_two_phase:
            logger.info(f"[EXTRACT] 📊 Two-phase extraction: {len(all_entities)} entities, triggering separate relation extraction")
            print(f"[EXTRACT] 📊 Two-phase: {len(all_entities)} entities, triggering relation extraction...")

            try:
                relation_result = await self._extract_relationships_separate(text, all_entities, domain)
                if relation_result and len(relation_result) > 0:
                    relationships.extend(relation_result)
                    logger.info(f"[EXTRACT] 📊 Two-phase extraction completed: added {len(relation_result)} relationships")
                    print(f"[EXTRACT] ✅ Two-phase: added {len(relation_result)} relationships")
                else:
                    logger.info(f"[EXTRACT] 📊 Two-phase extraction returned no additional relationships")
            except Exception as e:
                logger.warning(f"[EXTRACT] Two-phase relation extraction failed: {e}")
                print(f"[EXTRACT] ⚠️ Two-phase relation extraction failed: {e}")
        elif _skip_two_phase:
            print(f"[EXTRACT] ⏭️ Two-phase relation extraction skipped (EXTRACT_SKIP_TWO_PHASE=1)")
        
        # Calculate quality score using unified scorer
        quality_score = await calculate_unified_quality_score(
            l1_entities, l2_entities, l25_entities, relationships, domain
        )

        validation_passed = (not result_degraded) and len(l1_entities) > 0 and len(relationships) > 0
        extraction_quality_status = (
            "passed" if validation_passed and quality_score >= 0.6 else
            "partial" if len(l1_entities) > 0 or len(l2_entities) > 0 or len(relationships) > 0 else
            "failed"
        )
        validation_reason = None
        if result_degraded:
            validation_reason = "degraded extraction after validation issues"
        elif len(l1_entities) == 0:
            validation_reason = "no structured entities extracted"
        elif len(relationships) == 0:
            validation_reason = "no relationships extracted"
        elif quality_score < 0.6:
            validation_reason = f"low quality score ({quality_score:.2f})"

        # Update statistics
        self._call_count += 1
        self._total_time += processing_time / 1000

        result_metadata = {
            "document_type": (document_metadata or {}).get("document_type") if isinstance(document_metadata, dict) else None,
            "document_type_confidence": (document_metadata or {}).get("document_type_confidence") if isinstance(document_metadata, dict) else None,
            "secondary_types": (document_metadata or {}).get("secondary_types", []) if isinstance(document_metadata, dict) else [],
            "source_path": str(file_path) if file_path else (document_metadata or {}).get("source_path", ""),
            "matched_skills": (document_metadata or {}).get("matched_skills", []) if isinstance(document_metadata, dict) else [],
            "skill_stats": (document_metadata or {}).get("skill_stats", {}) if isinstance(document_metadata, dict) else {},
            "validation_passed": validation_passed,
            "extraction_quality_status": extraction_quality_status,
            "validation_reason": validation_reason,
        }
        result = UnifiedExtractionResult(
            l0_entities=l0_entities,
            l1_entities=l1_entities,
            l2_entities=l2_entities,
            l25_entities=l25_entities,
            l3_entities=l3_entities,
            relationships=relationships,
            raw_response=response_text,
            processing_time_ms=processing_time,
            token_usage={"estimated": len(response_text) // 4},  # Rough estimate
            quality_score=quality_score,
            degraded=result_degraded  # Phase 1: Track degraded mode
        )
        setattr(result, "metadata", result_metadata)
        # T2.1 dual-track: flag-gated OntologyPacket attachment, additive only.
        maybe_attach_ontology_packet(result)
        print(
            f"[EXTRACT][QUALITY] status={extraction_quality_status} "
            f"validation_passed={validation_passed} "
            f"quality={quality_score:.2f} "
            f"l1={len(l1_entities)} l2={len(l2_entities)} rels={len(relationships)}"
        )
        if validation_reason:
            print(f"[EXTRACT][QUALITY] reason={validation_reason}")

        logger.info(
            f"Unified extraction complete: "
            f"L0={len(l0_entities)}, L1={len(l1_entities)}, L2={len(l2_entities)}, "
            f"L2.5={len(l25_entities)}, L3={len(l3_entities)}, "
            f"Rels={len(relationships)}, time={processing_time:.1f}ms, quality={quality_score:.2f}, "
            f"retries={attempt}, degraded={result_degraded}, status={extraction_quality_status}"
        )

        return result
    
    async def _run_llm_extraction_loop(
        self,
        prompt: str,
        text: str,
        domain: Optional[str],
        use_context: bool,
        max_retries: int,
        enable_validation: bool,
        document_metadata: Optional[Dict[str, Any]],
    ):
        """Run the single LLM call with validation and retry.

        Extracted from ``extract_unified`` (2026-08-11). Returns the parsed
        entity lists plus state the caller still needs (response_text,
        degraded flag, final attempt, and the possibly-shortened text/prompt).
        """
        result_degraded = False

        prompt_metrics: Dict[str, Any] = {}

        llm_stats = {
            "total_calls": 0,
            "truncated_count": 0,
            "retry_count": 0,
            "parse_success_count": 0
        }
        
        for attempt in range(max_retries + 1):
            try:
                llm_result = await self._call_llm(prompt)
                llm_stats["total_calls"] += 1
                
                response_text = llm_result["response"]
                truncated = llm_result.get("truncated", False)
                parse_success = llm_result.get("parse_success", False)
                
                llm_stats["truncated_count"] += 1 if truncated else 0
                llm_stats["parse_success_count"] += 1 if parse_success else 0
                
                if truncated and not parse_success and attempt < max_retries:
                    logger.warning(f"[EXTRACT] Response truncated and parse failed. Retrying with shorter input (attempt {attempt + 1}/{max_retries + 1})")
                    print(f"[EXTRACT] ⚠️ Retrying due to truncation...")
                    
                    text = text[:int(len(text) * 0.7)]
                    prompt = self._build_unified_prompt(text, domain, use_context, document_metadata=document_metadata, metrics=prompt_metrics)
                    llm_stats["retry_count"] += 1
                    continue
                
                # Parse the unified response
                l0_entities, l1_entities, l2_entities, l25_entities, l3_entities, relationships = self._parse_unified_response(response_text)
                
                # Phase 1: Enhanced validation with graceful degradation
                if enable_validation:
                    validation_result = validate_extraction(
                        l1_entities, l2_entities, l25_entities, relationships,
                        use_pydantic=True
                    )
                    try:
                        from ontology_compiler.schemas.extraction_schema import validate_extraction_result
                        schema_validation = validate_extraction_result({
                            "l1_entities": l1_entities,
                            "l2_entities": l2_entities,
                            "l25_entities": l25_entities,
                            "relationships": relationships,
                        }, strict=False, document_type=(document_metadata or {}).get("document_type") if isinstance(document_metadata, dict) else None)
                        if not schema_validation.get("valid", True):
                            validation_result.setdefault("errors", []).extend(schema_validation.get("errors", []))
                            validation_result["valid"] = False
                        if schema_validation.get("warnings"):
                            validation_result.setdefault("warnings", []).extend(schema_validation.get("warnings"))
                    except Exception as schema_err:
                        logger.warning(f"[EXTRACT] Schema-aware validation skipped: {schema_err}")
                    semantic_result = validate_response_semantics(
                        l1_entities, l2_entities, l25_entities, l3_entities, relationships
                    )
                    if not semantic_result["valid"]:
                        validation_result.setdefault("errors", []).extend(
                            [str(issue["type"]) for issue in semantic_result.get("errors", [])]
                        )
                        validation_result["valid"] = False
                    if semantic_result.get("warnings"):
                        validation_result.setdefault("warnings", []).extend(
                            [str(w["type"]) for w in semantic_result["warnings"]]
                        )
                    
                    # Classify validation issues
                    has_critical_errors = len(validation_result.get("errors", [])) > 0
                    has_warnings = len(validation_result.get("warnings", [])) > 0
                    stats = validation_result.get("stats", {})
                    total_entities = sum(stats.get(k, 0) for k in ["l1_count", "l2_count", "l25_count"])
                    
                    # Degraded mode decision tree - faster recovery
                    if has_critical_errors:
                        # If we already have entities, proceed in degraded mode immediately to save time
                        if total_entities > 0:
                            logger.warning(
                                f"Validation failed with {total_entities} entities. "
                                f"Proceeding in DEGRADED mode to save time."
                            )
                            result_degraded = True
                            break
                            
                        if attempt < max_retries:
                            logger.warning(
                                f"Validation failed (attempt {attempt + 1}/{max_retries + 1}): "
                                f"{validation_result['errors'][:3]}"  # Log first 3 errors
                            )
                            # Add validation feedback to prompt for retry
                            prompt = add_validation_feedback(prompt, validation_result)
                            last_error = f"Validation failed: {validation_result['errors']}"
                            continue
                        else:
                            logger.error(
                                f"Validation failed after {max_retries + 1} attempts "
                                f"with 0 entities. Raising exception."
                            )
                            raise ValueError(f"Extraction validation failed: {validation_result['errors']}")
                    elif has_warnings:
                        logger.info(f"Validation passed with warnings: {validation_result['warnings']}")
                        result_degraded = False
                
                # Success - break retry loop
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Extraction failed (attempt {attempt + 1}): {e}")
                    last_error = str(e)
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                else:
                    logger.error(f"Extraction failed after {max_retries + 1} attempts: {e}")
                    raise

        return (
            l0_entities, l1_entities, l2_entities, l25_entities, l3_entities, relationships,
            response_text, result_degraded, attempt, text, prompt,
        )

    async def extract_all_layers(
        self,
        text: str,
        file_path: Optional[str] = None,
        domain: Optional[str] = None,
        use_context: bool = True,
        enable_validation: bool = True,
        max_retries: int = 3,
        fast_mode: bool = False,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> UnifiedExtractionResult:
        return await self.extract_unified(
            text=text,
            file_path=file_path,
            domain=domain,
            use_context=use_context,
            enable_validation=enable_validation,
            max_retries=max_retries,
            fast_mode=fast_mode,
            document_metadata=document_metadata or {}
        )
    
    async def _extract_with_chunking(
        self,
        text: str,
        file_path: Optional[Path] = None,
        domain: Optional[str] = None,
        use_context: bool = True,
        enable_validation: bool = True,
        max_retries: int = 1,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedExtractionResult:
        """Extract entities from large documents using smart chunking.

        Extracted to ``chunking_service.extract_with_chunking`` (2026-08-11);
        this wrapper keeps the original call sites unchanged. The explicit
        chunk strategy override (slide_based / section_based / hybrid) is
        forwarded when configured on the extractor.
        """
        strategy = getattr(self, "chunk_strategy", None) or ""
        cache_enabled = bool(os.environ.get("EXTRACT_CHUNK_CACHE", "1") != "0")
        return await extract_with_chunking(
            text=text,
            file_path=file_path,
            domain=domain,
            use_context=use_context,
            enable_validation=enable_validation,
            max_retries=max_retries,
            document_metadata=document_metadata,
            chunker=self.chunker,
            build_prompt=self._build_unified_prompt,
            call_llm=self._call_llm,
            parse_response=self._parse_unified_response,
            extract_relationships=self._extract_relationships_separate,
            result_cls=UnifiedExtractionResult,
            strategy=strategy,
            cache_enabled=cache_enabled,
        )

    def _normalize_entities(
        self,
        l0_entities: List[Dict[str, Any]],
        l1_entities: List[Dict[str, Any]],
        l2_entities: List[Dict[str, Any]],
        l25_entities: List[Dict[str, Any]],
        l3_entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], ...]:
        """
        Post-extraction normalization: auto-fill missing status fields and normalize entity names.
        
        This ensures compliance with quality requirements regardless of LLM behavior.
        """
        normalized_l0, normalized_l1, normalized_l2, normalized_l25, normalized_l3 = normalize_entities_layers(
            l0_entities, l1_entities, l2_entities, l25_entities, l3_entities
        )

        total = len(normalized_l0) + len(normalized_l1) + len(normalized_l2) + len(normalized_l25) + len(normalized_l3)
        logger.info(f"Entity normalization complete: processed {total} entities, applied naming conventions and status defaults")

        return normalized_l0, normalized_l1, normalized_l2, normalized_l25, normalized_l3
    
    def _build_unified_prompt(
        self,
        text: str,
        domain: Optional[str] = None,
        use_context: bool = True,
        skip_truncate: bool = False,
        document_metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the unified extraction prompt via layered composition.

        Extracted to ``PromptService.build_unified_prompt`` (2026-08-11);
        this wrapper keeps the original call sites unchanged. ``metrics``
        (optional) collects D-5 truncation/compression observability in-place.
        """
        return self._prompt_service.build_unified_prompt(
            text=text,
            domain=domain,
            use_context=use_context,
            skip_truncate=skip_truncate,
            document_metadata=document_metadata,
            metrics=metrics,
        )

    async def _quality_refine_if_needed(self, text: str, l2_entities: List[Dict[str, Any]], domain: Optional[str]) -> List[Dict[str, Any]]:
        """Refine L2 extraction when quality or quantity is insufficient."""
        if len(l2_entities) >= 10 and self.enable_refinement:
            return l2_entities
        if len(l2_entities) >= self.refinement_threshold and not self.enable_refinement:
            return l2_entities
        return await self._refine_l2(text, l2_entities, domain or "GENERAL")
    
    async def _call_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Make async LLM call with tracing, smart routing, and JSON validation with retries.

        Extracted to ``ontology_compiler.services.llm_call.call_llm`` (2026-08-11);
        this wrapper keeps the original call sites and trace step names unchanged.
        """
        return await call_llm(
            llm=self.llm,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            temperature=self.temperature,
            timeout=self.timeout,
            current_run_id=self._current_run_id,
            tracer=self.tracer,
            get_metadata=self.get_metadata,
            prompt=prompt,
            **kwargs,
        )
    def _parse_unified_response(
        self, 
        response: str
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """
        Parse the unified LLM response into three entity lists + relationships.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (l1_entities, l2_entities, l25_entities, relationships)
        """
        return parse_unified_response(
            response,
            infer_entity_class=infer_entity_class,
            assign_meta_ids=self._assign_meta_ids,
        )
    def _assign_meta_ids(
        self, 
        entities: List[Dict], 
        prefix: str
    ) -> List[Dict]:
        """Assign meta_ids to entities that don't have them."""
        for i, entity in enumerate(entities):
            if not entity.get("meta_id"):
                entity["meta_id"] = f"{prefix}-{i+1:03d}"
        return entities
    
    def _apply_skill_governance(
        self,
        text: str,
        domain: Optional[str],
        l1_entities: List[Dict],
        l2_entities: List[Dict],
        l25_entities: List[Dict],
        relationships: List[Dict],
    ) -> Dict[str, Any]:
        """根据已激活 skill 的 governance 字段对抽取结果做后置校验。

        Extracted to ``governance_service.apply_skill_governance`` (2026-08-11);
        this wrapper keeps the original call sites unchanged.
        """
        return apply_skill_governance(
            text=text,
            domain=domain,
            l1_entities=l1_entities,
            l2_entities=l2_entities,
            l25_entities=l25_entities,
            relationships=relationships,
            skill_loader=self.skill_loader,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
            "avg_processing_time_ms": (self._total_time / self._call_count * 1000) if self._call_count > 0 else 0,
            "total_time_seconds": self._total_time,
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self._call_count = 0
        self._total_tokens = 0
        self._total_time = 0.0


# Convenience function for direct usage
async def extract_unified(
    text: str,
    domain: Optional[str] = None,
    use_context: bool = True
) -> UnifiedExtractionResult:
    """
    Quick unified extraction function.
    
    Args:
        text: Document text
        domain: Optional domain hint
        use_context: Whether to use master ontology context
        
    Returns:
        UnifiedExtractionResult
    """
    extractor = TrueUnifiedExtractor()
    return await extractor.extract_all_layers(text, domain=domain, use_context=use_context)


# Backward-compatible aliases for deprecated modules
class ExtractionStrategyService(TrueUnifiedExtractor):
    """Backward-compatible alias for the unified extractor.
    
    This class exists solely for backward compatibility with legacy code
    that imports from `ontology_compiler.services.extraction_strategy_service`.
    New code should use `TrueUnifiedExtractor` directly.
    """
    pass


class FileSelector(TrueUnifiedExtractor):
    """Backward-compatible alias for the unified extractor.
    
    This class exists solely for backward compatibility with legacy code
    that imports from `ontology_compiler.services.file_selector`.
    New code should use `TrueUnifiedExtractor` directly.
    """
    pass


__all__ = [
    "UnifiedExtractionResult",
    "DocumentAnalyzer",
    "PromptStrategy",
    "PromptService",
    "TrueUnifiedExtractor",
    "ExtractionStrategyService",
    "FileSelector",
    "extract_unified",
]
