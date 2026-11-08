"""Continuity + artifact registration extracted from MainPipeline (P0-3 batch 15).

Run-continuity lifecycle, processed-file filtering and stage-artifact
registration, moved verbatim from MainPipeline methods using the established
``pipeline``-first-parameter pattern (same as runflow, status_quality,
stage_plumbing, trace_provenance, regression_metadata, result_helpers).

Contents:
- :func:`begin_run_continuity` — from ``_begin_run_continuity``
- :func:`load_regression_baseline_if_needed` — from
  ``_load_regression_baseline_if_needed``
- :func:`load_regression_baseline` — from ``_load_regression_baseline``
- :func:`end_run_continuity` — from ``_end_run_continuity``
- :func:`filter_unprocessed_files` — from ``_filter_unprocessed_files``
- :func:`filter_single_processed_file` — from
  ``_filter_single_processed_file``
- :func:`filter_processed_directory_files` — from
  ``_filter_processed_directory_files``
- :func:`register_stage_artifacts` — from ``_register_stage_artifacts``
- :func:`register_stage_bundle` — from ``_register_stage_bundle``
- :func:`record_stage_artifact_outputs` — from
  ``_record_stage_artifact_outputs``
- :func:`mark_source_file_processed_if_needed` — from
  ``_mark_source_file_processed_if_needed``

Behavior stays locked by tests/pipeline/test_main_pipeline_characterization.py
(``_begin_run_continuity`` / ``_end_run_continuity`` /
``_register_stage_artifacts`` are patched at instance level in the run-flow
tests; the MainPipeline delegates preserve those seams).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline
    from .pipeline_models import PipelineStage
    from .stages import StageResult

logger = logging.getLogger(__name__)


def begin_run_continuity(pipeline: "MainPipeline") -> None:
    """Begin run-continuity tracking and capture extracted metadata."""
    pipeline._load_regression_baseline_if_needed()
    try:
        # B4 (2026-08-15): file traversal + service call moved to the
        # continuity use case; this function keeps baseline/metadata wiring.
        from src.application.use_cases.pipeline_continuity_use_case import (
            begin_run,
            collect_input_files,
        )
        from . import regression_metadata

        input_files = collect_input_files(pipeline.context.source_path)
        begin_run(
            pipeline._continuity_service,
            run_id=pipeline._run_id,
            session_id=pipeline._session_id,
            input_files=input_files,
        )
        # P0-3 batch 1: metadata capture/persistence moved verbatim to
        # src/pipeline/regression_metadata.py (pipeline-first-parameter
        # pattern); the returned dict keeps self._extracted_metadata in
        # sync for _store_classification_metrics.
        pipeline._extracted_metadata = regression_metadata.capture_and_persist_extracted_metadata(pipeline)
        regression_metadata.persist_regression_manifest(pipeline)
        logger.info(f"📋 Run continuity started: {pipeline._run_id}, {len(input_files)} files")
    except Exception as e:
        logger.warning(f"Failed to begin run continuity: {e}")


def load_regression_baseline_if_needed(pipeline: "MainPipeline") -> None:
    """Load the regression baseline into context when in regression mode."""
    try:
        if pipeline.context.regression_mode:
            pipeline._regression_baseline = load_regression_baseline(pipeline)
            if pipeline._regression_baseline:
                pipeline.context.metadata["regression_baseline"] = pipeline._regression_baseline
    except Exception as e:
        logger.warning(f"Failed to load regression baseline: {e}")


def load_regression_baseline(pipeline: "MainPipeline") -> Dict[str, Any]:
    """Build the regression baseline from the sample manifest expectations."""
    baseline: Dict[str, Any] = {}
    try:
        manifest = pipeline.context.metadata.get("regression_manifest", {}) or {}
        expected = manifest.get("expected", {}) if isinstance(manifest, dict) else {}
        if isinstance(expected, dict):
            baseline.update(expected)
    except Exception:
        pass
    return baseline


def end_run_continuity(pipeline: "MainPipeline", status: str) -> None:
    """Finalize run-continuity tracking for the run."""
    try:
        # B4 (2026-08-15): service call + artifact-path mapping moved to
        # the continuity use case; this function keeps only state wiring.
        from src.application.use_cases.pipeline_continuity_use_case import (
            end_run,
            build_artifact_paths,
        )

        end_run(
            pipeline._continuity_service,
            run_id=pipeline._run_id,
            status=status,
            output_path=pipeline.context.output_dir,
            entity_count=pipeline._stats.entity_count,
            relation_count=pipeline._stats.relation_count,
            llm_call_count=pipeline._stats.llm_calls,
            errors=pipeline._errors,
            warnings=pipeline._warnings,
            artifact_paths=build_artifact_paths(pipeline._output_files),
            classification_metrics=getattr(pipeline.context, "metadata", {}).get("classification_metrics", {}),
        )
        logger.info(f"📋 Run continuity ended: {pipeline._run_id}, status={status}")
    except Exception as e:
        logger.warning(f"Failed to end run continuity: {e}")


def filter_unprocessed_files(pipeline: "MainPipeline") -> None:
    """Skip files already processed by a previous run."""
    try:
        source_path = Path(pipeline.context.source_path)
        if source_path.is_file():
            filter_single_processed_file(pipeline, source_path)
        elif source_path.is_dir():
            filter_processed_directory_files(pipeline, source_path)
    except Exception as e:
        logger.debug(f"File filtering skipped: {e}")


def filter_single_processed_file(pipeline: "MainPipeline", source_path: Path) -> None:
    """Record a skip marker when a single file was already processed."""
    if not pipeline._continuity_service.should_process_file(source_path):
        logger.info(f"⏭️  Skipping already processed file: {source_path}")
        pipeline.context.metadata["skipped_files"] = [str(source_path)]


def filter_processed_directory_files(pipeline: "MainPipeline", source_path: Path) -> None:
    """Record skip markers for already-processed files in a directory."""
    skipped_files = []
    for ext in ["*.pdf", "*.docx", "*.doc", "*.md", "*.txt"]:
        for file_path in source_path.rglob(ext):
            if not pipeline._continuity_service.should_process_file(file_path):
                skipped_files.append(str(file_path))
    if skipped_files:
        logger.info(f"⏭️  Skipping {len(skipped_files)} already processed files")
        pipeline.context.metadata["skipped_files"] = skipped_files


def register_stage_artifacts(pipeline: "MainPipeline", stage: "PipelineStage", stage_result: "StageResult") -> None:
    """Register stage artifacts to the artifact index + continuity service."""
    try:
        stage_data = stage_result.data or {}
        output_files = stage_result.output_files if hasattr(stage_result, "output_files") else []

        bundle_path = register_stage_bundle(pipeline, stage, stage_result, stage_data, output_files)
        record_stage_artifact_outputs(pipeline, stage, stage_result, stage_data, output_files)
        mark_source_file_processed_if_needed(pipeline, stage_result)
    except Exception as e:
        logger.debug(f"Artifact registration skipped: {e}")


def register_stage_bundle(
    pipeline: "MainPipeline",
    stage: "PipelineStage",
    stage_result: "StageResult",
    stage_data: Dict[str, Any],
    output_files: list,
) -> str:
    """Register an ArtifactBundle and return its path."""
    from .artifact_registry import ArtifactBundle
    from .versioned_cache import resolve_version_bundle_from_context

    stage_bundle = ArtifactBundle(
        run_id=pipeline._run_id,
        stage=stage.value,
        artifact_root=str(pipeline._run_paths.artifact_root),
        audit_root=str(pipeline._run_paths.audit_root),
        regression_root=str(pipeline._run_paths.regression_root),
        items={Path(output_file).name: output_file for output_file in output_files[:10]},
        metadata={
            "success": stage_result.success,
            "entity_count": stage_data.get("entity_count", 0),
            "relation_count": stage_data.get("relation_count", 0),
            "stage": stage.value,
            # C-4 cross-run lookup signals: source_hash + version_bundle must
            # match exactly before StageDecision.REUSE_ARTIFACT fires.
            "source_hash": pipeline._run_id,
            "source_path": pipeline.context.source_path,
            "version_bundle": resolve_version_bundle_from_context(pipeline.context).to_dict(),
            # Cache the StageResult data shape so a future REUSE_ARTIFACT
            # can synthesise the exact same downstream contract without
            # re-running the stage handler (see _load_stage_result_from_bundle).
            "stage_result_data": dict(stage_data or {}),
            "stage_result_message": stage_result.message or "",
            "stage_result_elapsed_seconds": getattr(stage_result, "elapsed_seconds", 0.0) or 0.0,
            "stage_result_fragment_path": getattr(stage_result, "fragment_path", "") or "",
            "stage_result_run_dir": getattr(stage_result, "run_dir", "") or "",
            "stage_result_output_files": list(output_files or [])[:50],
            "stage_result_artifact_paths": list(getattr(stage_result, "artifact_paths", None) or [])[:50],
        },
    )
    bundle_path = pipeline._artifact_registry.register_bundle(stage_bundle)
    pipeline._output_files.append(bundle_path)
    pipeline.context.metadata.setdefault("artifact_bundles", []).append(bundle_path)
    return bundle_path


def record_stage_artifact_outputs(
    pipeline: "MainPipeline",
    stage: "PipelineStage",
    stage_result: "StageResult",
    stage_data: Dict[str, Any],
    output_files: list,
) -> None:
    """Register each stage output file with the continuity service."""
    for output_file in output_files[:10]:
        artifact_id = f"{stage.value}-{Path(output_file).name}"
        pipeline._continuity_service.register_artifact(
            artifact_id=artifact_id,
            artifact_type=stage.value,
            run_id=pipeline._run_id,
            source_file=pipeline.context.source_path,
            file_path=output_file,
            metadata={
                "stage": stage.value,
                "success": stage_result.success,
                "entity_count": stage_data.get("entity_count", 0),
                "relation_count": stage_data.get("relation_count", 0),
            },
        )


def mark_source_file_processed_if_needed(pipeline: "MainPipeline", stage_result: "StageResult") -> None:
    """Record the source file as processed after a successful stage."""
    if not stage_result.success:
        return
    source_path = Path(pipeline.context.source_path)
    if source_path.is_file():
        pipeline._continuity_service.record_file_processed(
            file_path=source_path,
            run_id=pipeline._run_id,
            status="success",
            output_path=pipeline.context.output_dir,
        )
