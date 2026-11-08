"""Regression/continuity metadata persistence extracted from MainPipeline.

P0-3 batch 1 of the main_pipeline LOC reduction (docs/
OPTIMIZATION_PRIORITY_ACTION_LIST.md v2.0): the extracted-metadata capture and
the regression-manifest persistence moved verbatim from MainPipeline methods
using the established ``pipeline``-first-parameter pattern (same as
``stage_execution_core.run_pipeline_stages`` and
``asset_packet.generate_asset_packet``).

Contents:
- :func:`collect_continuity_sample_text` — raw text sampling for extraction
- :func:`capture_and_persist_extracted_metadata` — extract + project + persist
- :func:`persist_extracted_metadata` — write extracted_metadata.json artifact
- :func:`persist_regression_manifest` — write regression_manifest.json artifact

Behavior stays locked by tests/pipeline/test_main_pipeline_characterization.py,
which exercises the pipeline through ``_begin_run_continuity``; the moved
methods had no direct callers outside MainPipeline (verified 2026-09-09).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from .stages.common import _write_json, project_metadata_to_context
from .metadata_enrichment import extract_metadata

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline

logger = logging.getLogger(__name__)


def collect_continuity_sample_text(pipeline: "MainPipeline") -> str:
    """Collect the raw source text used for continuity metadata extraction.

    Moved verbatim from ``MainPipeline._collect_continuity_sample_text``.
    Reads the file directly, or concatenates ``*.md`` / ``*.txt`` files when
    the source is a directory. Never raises: failures degrade to ``""``.
    """
    try:
        source_path = Path(pipeline.context.source_path)
        if source_path.is_file():
            return source_path.read_text(encoding="utf-8", errors="replace")
        if source_path.is_dir():
            sample_text = []
            for ext in ["*.md", "*.txt"]:
                for file_path in source_path.rglob(ext):
                    sample_text.append(file_path.read_text(encoding="utf-8", errors="replace"))
            return "\n".join(sample_text)
    except Exception:
        return ""
    return ""


def capture_and_persist_extracted_metadata(pipeline: "MainPipeline") -> Dict[str, Any]:
    """Extract domain metadata from the source sample and persist it.

    Moved verbatim from ``MainPipeline._capture_and_persist_extracted_metadata``.
    Returns the extracted metadata dict (empty when no sample text), so the
    orchestrator can keep ``pipeline._extracted_metadata`` in sync for
    ``_store_classification_metrics``.
    """
    sample_text = collect_continuity_sample_text(pipeline)
    extracted = extract_metadata(sample_text).to_dict() if sample_text else {}
    if not extracted:
        return extracted
    pipeline.context.metadata["extracted_metadata"] = extracted
    project_metadata_to_context(extracted, pipeline.context.metadata)
    persist_extracted_metadata(pipeline, extracted)
    return extracted


def persist_extracted_metadata(pipeline: "MainPipeline", extracted: Dict[str, Any]) -> None:
    """Write ``extracted_metadata.json`` under the run's artifact/regression root.

    Moved verbatim from ``MainPipeline._persist_extracted_metadata``.
    """
    metadata_path = pipeline._run_paths.regression_root / "extracted_metadata.json" if pipeline.context.regression_mode else pipeline._run_paths.artifact_root / "extracted_metadata.json"
    _write_json(metadata_path, extracted)
    pipeline.context.metadata["extracted_metadata_path"] = str(metadata_path)


def persist_regression_manifest(pipeline: "MainPipeline") -> None:
    """Attach the sample's regression manifest to the run artifacts.

    Moved verbatim from ``MainPipeline._persist_regression_manifest``.
    """
    sample_manifest = pipeline.context.metadata.get("regression_manifest")
    if not sample_manifest:
        return
    manifest_path = pipeline._run_paths.regression_root / "regression_manifest.json"
    _write_json(manifest_path, sample_manifest)
    pipeline.context.metadata["regression_manifest_path"] = str(manifest_path)
    logger.info(f"📎 Regression manifest attached: {manifest_path}")
