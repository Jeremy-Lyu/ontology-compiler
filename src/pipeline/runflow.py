"""Runflow helpers extracted from MainPipeline (P0-3 batch 12).

Run-start / contract-validation / stage-resolution / classification-metrics
wiring, moved verbatim from MainPipeline methods using the established
``pipeline``-first-parameter pattern (same as status_quality, stage_plumbing,
trace_provenance, regression_metadata, result_helpers).

Contents:
- :func:`log_pipeline_start` — D-4 LLM-counter reset + run banner
- :func:`ensure_contract_validation` — pre-run contract gate decision
- :func:`build_failed_contract_result` — failure result assembly
- :func:`resolve_stage_enums` — stage-name → enum resolution
- :func:`collect_classification_metrics` — B2 use-case wiring
- :func:`store_classification_metrics` — stats/metadata wiring

Behavior stays locked by tests/pipeline/test_main_pipeline_characterization.py
(via the end-to-end run paths, the contract-failure path that patches
``_run_contract_validation_chain`` at instance level, and the snapshot shape
tests).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import (
        MainPipeline,
        PipelineExecutionResult,
        PipelineStage,
    )

logger = logging.getLogger(__name__)


def log_pipeline_start(pipeline: "MainPipeline") -> None:
    """Reset LLM counters and emit the run banner."""
    # D-4 (2026-08-15): reset LLM counters before each run so manifest
    # llm_stats reflect this run only (no cross-run bleed).
    try:
        from src.application.use_cases.llm_observability_use_case import reset_llm_monitor
        from src.application.services.llm_monitor import get_llm_monitor
        reset_llm_monitor(get_llm_monitor())
    except Exception:
        pass
    logger.info(f"🚀 Starting Main Pipeline: {pipeline._run_id}")
    logger.info(f"Source: {pipeline.context.source_path}, Domain: {pipeline.context.domain}")
    logger.info(f"Execution mode: {pipeline.context.execution_mode}")


def ensure_contract_validation(pipeline: "MainPipeline") -> bool:
    """Run the pre-execution contract gate; hard-stop only in production."""
    contract_valid = pipeline._run_contract_validation_chain()
    if contract_valid:
        return True
    pipeline._errors.append("Contract validation chain failed")
    from .main_pipeline import PipelineExecutionMode

    if pipeline._execution_mode == PipelineExecutionMode.PRODUCTION:
        logger.error("🔴 Contract validation chain failed, stopping pipeline")
        return False
    logger.warning("🟡 Contract validation failed, continuing in reduced execution mode")
    pipeline._execution_reason = "contract_validation_failed"
    pipeline._warnings.append("Contract validation failed; continuing in reduced execution mode")
    return True


def build_failed_contract_result(pipeline: "MainPipeline") -> "PipelineExecutionResult":
    """Assemble the failure result for a failed pre-run contract gate."""
    from .main_pipeline import (
        PipelineExecutionResult,
        PipelineQualityStatus,
        PipelineStatus,
    )

    return PipelineExecutionResult(
        run_id=pipeline._run_id,
        status=PipelineStatus.FAILED,
        context=pipeline.context,
        stats=pipeline._stats,
        stage_results=pipeline._stage_results,
        output_files=pipeline._output_files,
        errors=pipeline._errors,
        warnings=pipeline._warnings,
        publishable=False,
        execution_mode=pipeline._execution_mode.value,
        execution_reason=pipeline._execution_reason,
        quality_status=PipelineQualityStatus.FAILURE,
        quality_reason="contract validation failed",
        validation_passed=False,
        quality_score=0.0,
    )


def resolve_stage_enums(pipeline: "MainPipeline", stages: Optional[List[str]]) -> List["PipelineStage"]:
    """Resolve requested stage names to enums (default: full order)."""
    from .main_pipeline import PipelineStage

    if stages:
        stage_enums = [PipelineStage(s) for s in stages if s.upper() in PipelineStage.__members__]
        logger.info(f"Executing specific stages: {[s.value for s in stage_enums]}")
        return stage_enums
    return pipeline._stage_order


def collect_classification_metrics(pipeline: "MainPipeline") -> None:
    """Collect the classification summary into context/stats state."""
    # B2 (2026-08-15): collection + summarization migrated to the
    # classification_metrics use case; this function only wires the summary
    # back into context/stats state.
    try:
        from src.application.use_cases.classification_metrics_use_case import (
            summarize_classification,
        )
        from src.application.services.classification_service import get_classification_service

        classification_summary = summarize_classification(
            pipeline._stage_results,
            get_classification_service=get_classification_service,
        )
        if classification_summary is None:
            return
        pipeline.context.metadata.setdefault("classification_metrics", {}).update(classification_summary)
        store_classification_metrics(pipeline, classification_summary)
    except Exception as e:
        logger.warning(f"Classification metrics collection skipped: {e}")


def store_classification_metrics(pipeline: "MainPipeline", classification_summary) -> None:
    """Wire a classification summary into stats + context metadata."""
    pipeline._stats.metadata = getattr(pipeline._stats, "metadata", {})
    pipeline._stats.metadata["classification_metrics"] = classification_summary
    if pipeline._extracted_metadata:
        pipeline.context.metadata.setdefault("extracted_metadata", pipeline._extracted_metadata)
        pipeline._stats.metadata["extracted_metadata"] = pipeline._extracted_metadata
