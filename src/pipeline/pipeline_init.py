"""Pipeline construction state extracted from MainPipeline (P0-3 batch 17).

The ``__init__`` sub-steps, moved verbatim from MainPipeline methods using the
established ``pipeline``-first-parameter pattern (same as runflow,
status_quality, stage_plumbing, trace_provenance, regression_metadata).

Contents:
- :func:`normalize_output_dir` — from ``_normalize_output_dir`` (static)
- :func:`generate_run_id` — from ``_generate_run_id`` (static)
- :func:`compute_file_hash` — from ``_compute_file_hash`` (static)
- :func:`determine_execution_mode` — from ``_determine_execution_mode``
- :func:`set_execution_mode` — from ``_set_execution_mode``
- :func:`init_context_state` — context/run-id/output-dir/hash/paths/metadata
- :func:`init_run_state` — stage results/stats/errors/outputs/baseline
- :func:`init_services` — bridge/continuity/asset/retrieval/result services
- :func:`init_execution_state` — session id / artifact root / execution mode /
  extracted metadata / governance arming

Behavior stays locked by tests/pipeline/test_main_pipeline_characterization.py
(``_set_execution_mode`` is patched at instance level in the guard-failure
tests; the MainPipeline delegate preserves that seam).
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline
    from .pipeline_models import PipelineExecutionMode

logger = logging.getLogger(__name__)


def normalize_output_dir(output_dir: str, run_id: str) -> str:
    """Normalize the run output dir (collapse nested files/0000 segments)."""
    path = Path(output_dir)
    parts = list(path.parts)
    if parts.count("files") > 1:
        first_files_idx = parts.index("files")
        path = Path(*parts[:first_files_idx])
    if "0000" in parts and parts.count("0000") > 1:
        first_zero_idx = parts.index("0000")
        path = Path(*parts[:first_zero_idx])
    if path.name != run_id:
        path = path / run_id
    return str(path)


def generate_run_id() -> str:
    """Generate a timestamped run id."""
    return f"MAIN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def compute_file_hash(source_path: str) -> str:
    """Compute the sha256 of a source file ("" for dirs/missing)."""
    path = Path(source_path)
    if not path.exists() or not path.is_file():
        return ""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def determine_execution_mode(pipeline: "MainPipeline") -> "PipelineExecutionMode":
    """Determine the execution mode from context + governance flags."""
    from .pipeline_models import PipelineExecutionMode

    if pipeline.context.regression_mode:
        return PipelineExecutionMode.OFFLINE
    # C-1: respect governance.degraded_offline flag (from CLI --degraded-offline
    # or env var LLM_OFFLINE_MODE). This forces LLM-dependent stages to use
    # rule-only code paths regardless of other switches.
    gov = pipeline.context.metadata.get("governance", {}) if isinstance(pipeline.context.metadata, dict) else {}
    if gov.get("degraded_offline") or os.environ.get("LLM_OFFLINE_MODE", "0") not in ("0", "", "false", "False"):
        return PipelineExecutionMode.DEGRADED
    if pipeline.context.fast_mode and pipeline.context.enable_llm_conflict_resolver is False:
        return PipelineExecutionMode.DEGRADED
    return PipelineExecutionMode.PRODUCTION


def set_execution_mode(pipeline: "MainPipeline", mode: "PipelineExecutionMode", reason: str = "") -> None:
    """Apply the execution mode to pipeline state + context metadata."""
    pipeline._execution_mode = mode
    pipeline.context.execution_mode = mode.value
    pipeline.context.execution_reason = reason
    pipeline.context.metadata["execution_mode"] = mode.value
    if reason:
        pipeline.context.metadata["execution_reason"] = reason
    pipeline._execution_reason = reason


def init_context_state(pipeline: "MainPipeline", context) -> None:
    """Wire context identity: run id, output dir, hash, paths, metadata."""
    from .pipeline_models import PipelineStats
    from .run_context import build_pipeline_run_paths

    pipeline.context = context
    pipeline._raw_context_output_dir = context.output_dir
    pipeline._run_id = context.run_id or generate_run_id()
    pipeline.context.run_id = pipeline._run_id
    pipeline.context.output_dir = normalize_output_dir(
        context.output_dir or str(_default_pipeline_dir() / pipeline._run_id),
        pipeline._run_id,
    )
    if not pipeline.context.source_hash:
        pipeline.context.source_hash = compute_file_hash(pipeline.context.source_path)
    pipeline.context.metadata.setdefault("source_hash", pipeline.context.source_hash)
    pipeline._run_paths = build_pipeline_run_paths(pipeline.context.output_dir, pipeline._run_id)
    pipeline.context.metadata.setdefault("run_paths", {
        "run_root": str(pipeline._run_paths.run_root),
        "stage_root": str(pipeline._run_paths.stage_root),
        "file_root": str(pipeline._run_paths.file_root),
        "artifact_root": str(pipeline._run_paths.artifact_root),
        "audit_root": str(pipeline._run_paths.audit_root),
        "regression_root": str(pipeline._run_paths.regression_root),
    })
    pipeline.context.metadata.setdefault("regression_mode", pipeline.context.regression_mode)
    pipeline.context.metadata.setdefault("execution_mode", pipeline.context.execution_mode)
    pipeline.context.metadata.setdefault("source_hash", pipeline.context.source_hash)
    pipeline.context.metadata.setdefault("schema_version", pipeline.context.schema_version or pipeline.context.metadata.get("schema_version", ""))
    pipeline.context.metadata.setdefault("skill_version", pipeline.context.skill_version or pipeline.context.metadata.get("skill_version", ""))
    pipeline.context.metadata.setdefault("prompt_version", pipeline.context.prompt_version or pipeline.context.metadata.get("prompt_version", ""))
    if pipeline._raw_context_output_dir != pipeline.context.output_dir:
        logger.info(
            "[DEBUG-PATH] MainPipeline.__init__ run_id=%s raw_output_dir=%s normalized_output_dir=%s",
            pipeline._run_id,
            pipeline._raw_context_output_dir,
            pipeline.context.output_dir,
        )
    pipeline._stats = PipelineStats()


def init_run_state(pipeline: "MainPipeline") -> None:
    """Initialize per-run mutable state + the fixed stage order."""
    from .pipeline_models import PipelineStage
    from .stages import StageResult  # noqa: F401  (kept for parity with original)

    pipeline._stage_results: dict[str, StageResult] = {}
    pipeline._errors: list[str] = []
    pipeline._warnings: list[str] = []
    pipeline._output_files: list[str] = []
    pipeline._regression_baseline: dict[str, object] = {}

    pipeline._stage_order = [
        PipelineStage.PREFLIGHT,
        PipelineStage.EXTRACT,
        PipelineStage.MERGE,
        PipelineStage.DEDUP,
        PipelineStage.GRAPH_FUSION,
        PipelineStage.RELATIONSHIP,
        PipelineStage.AUDIT,
        PipelineStage.DCS_AUDIT,
        PipelineStage.GUARD,
        PipelineStage.EMERGENT_REASONING,
        PipelineStage.EXPORT,
        PipelineStage.DOCUMENT_REPORT,
        PipelineStage.SUMMARIZE,
    ]

    pipeline._skill_contract_validator = None
    pipeline._active_skills: list[str] = []


def init_services(pipeline: "MainPipeline") -> None:
    """Acquire the service singletons used across stages."""
    from ontology_compiler.services.asset_index_service import get_asset_index_service
    from ontology_compiler.services.asset_packet_service import get_asset_packet_service
    from ontology_compiler.services.data_continuity import get_continuity_service
    from src.infrastructure.services.pipeline_result_artifact_service import (
        get_pipeline_result_artifact_service,
    )
    from src.infrastructure.services.retrieval_asset_packet_service import (
        get_retrieval_asset_packet_service,
    )
    from src.pipeline.skill_integration import get_skill_pipeline_bridge
    from .artifact_registry import ArtifactRegistry

    pipeline._skill_pipeline_bridge = get_skill_pipeline_bridge()
    pipeline._continuity_service = get_continuity_service()
    pipeline._asset_packet_service = get_asset_packet_service()
    pipeline._asset_index_service = get_asset_index_service()
    pipeline._retrieval_asset_packet_service = get_retrieval_asset_packet_service()
    pipeline._pipeline_result_artifact_service = get_pipeline_result_artifact_service()
    pipeline._artifact_registry = ArtifactRegistry(str(pipeline._run_paths.artifact_root))


def init_execution_state(pipeline: "MainPipeline") -> None:
    """Initialize session/execution-mode state and arm governance."""
    from . import governance_arming

    pipeline._session_id = f"SESSION-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{pipeline._run_id[-4:]}"
    pipeline._regression_artifact_root = pipeline._run_paths.regression_root if pipeline.context.regression_mode else pipeline._run_paths.artifact_root
    pipeline._execution_mode = determine_execution_mode(pipeline)
    set_execution_mode(pipeline, pipeline._execution_mode)
    pipeline._extracted_metadata: dict[str, object] = {}

    # ============================================================
    # C-1: LLM HARD BUDGET — FINAL DEFENSE LINE
    # CLI / run_batch should have armed the gateway already, but as
    # a last-resort safety net we re-arm here from context metadata
    # or env vars. If the caller already set a policy, the values
    # here will *tighten* (not relax) existing caps (same shape
    # overwrites the previous policy — this is intentionally a
    # stronger constraint, not a weaker one).
    # ============================================================
    pipeline._arm_governance_from_context()


def _default_pipeline_dir():
    from .stages import DEFAULT_PIPELINE_DIR

    return DEFAULT_PIPELINE_DIR
