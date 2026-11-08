"""Trace/provenance assembly extracted from MainPipeline (P0-3 batch 6).

Canonical-model builders moved verbatim from MainPipeline methods using the
established ``pipeline``-first-parameter pattern (same as result_helpers,
stage_execution_core, regression_metadata). These builders assemble
``TraceContext`` / provenance dicts / run metadata from the live pipeline
state; they own no execution logic.

Contents:
- :func:`build_trace_context` — from ``_build_trace_context``
- :func:`build_provenance` — from ``_build_provenance``
- :func:`build_run_metadata` — from ``_build_run_metadata``
- :func:`schema_version` / :func:`skill_version` / :func:`prompt_version` —
  from the ``_*_version`` accessors

Behavior stays locked by tests/pipeline/test_main_pipeline_characterization.py
(via the bridge + manifest + pipeline-result paths) and by the
batch 1–5 refactoring coverage.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from src.domain.models.core import TraceContext

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline


def schema_version(pipeline: "MainPipeline") -> str:
    """Resolved schema version for provenance/metadata blocks."""
    return pipeline.context.schema_version or pipeline.context.metadata.get("schema_version", "")


def skill_version(pipeline: "MainPipeline") -> str:
    """Resolved skill version for provenance/metadata blocks."""
    return pipeline.context.skill_version or pipeline.context.metadata.get("skill_version", "")


def prompt_version(pipeline: "MainPipeline") -> str:
    """Resolved prompt version for provenance/metadata blocks."""
    return pipeline.context.prompt_version or pipeline.context.metadata.get("prompt_version", "")


def build_run_metadata(pipeline: "MainPipeline") -> Dict[str, Any]:
    """Run-scoped metadata shared by trace + envelope payloads."""
    return {
        "domain": pipeline.context.domain,
        "source_path": pipeline.context.source_path,
        "source_hash": pipeline.context.source_hash,
        "schema_version": schema_version(pipeline),
        "skill_version": skill_version(pipeline),
        "prompt_version": prompt_version(pipeline),
    }


def build_provenance(pipeline: "MainPipeline") -> Dict[str, Any]:
    """Provenance dict attached to core results and envelopes."""
    return {
        "source_path": pipeline.context.source_path,
        "source_hash": pipeline.context.source_hash,
        "domain": pipeline.context.domain,
        "schema_version": schema_version(pipeline),
        "skill_version": skill_version(pipeline),
        "prompt_version": prompt_version(pipeline),
    }


def build_trace_context(
    pipeline: "MainPipeline",
    stage_id: str = "",
    task_id: str = "",
    artifact_id: str = "",
) -> TraceContext:
    """Assemble a ``TraceContext`` bound to this run."""
    return TraceContext(
        run_id=pipeline._run_id,
        session_id=pipeline._session_id,
        stage_id=stage_id,
        task_id=task_id,
        artifact_id=artifact_id,
        document_id=Path(pipeline.context.source_path).name if pipeline.context.source_path else "",
        metadata=build_run_metadata(pipeline),
    )
