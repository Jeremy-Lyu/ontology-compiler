"""Pipeline run-continuity use case (B4, 2026-08-15).

Extracted from ``MainPipeline._begin_run_continuity`` /
``_end_run_continuity`` / ``_collect_input_files_for_continuity`` /
``_build_artifact_paths_for_continuity`` so MainPipeline stops coupling
directly to the concrete ``DataContinuityService`` signature for these steps.

This module owns:
- ``collect_input_files``: enumerate source inputs for the run manifest
  (single file, or md/txt/pdf/docx/doc under a folder).
- ``build_artifact_paths``: map output-file basenames to paths for the run
  manifest's artifact index.
- ``begin_run`` / ``end_run``: thin orchestration over the injected service
  (begin manifest + fingerprint, end manifest + stats).

``MainPipeline`` keeps only the state wiring around these steps (baseline /
extracted-metadata capture, logging) and delegates pure traversal + service
calls here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_FILE_EXTS = ("*.pdf", "*.docx", "*.doc", "*.md", "*.txt")


def collect_input_files(source_path: str) -> List[str]:
    """Enumerate the source inputs for a run manifest.

    A single file yields that file; a folder yields every supported document
    under it (recursive). Mirrors the previous MainPipeline method exactly.
    """
    input_files: List[str] = []
    path = Path(source_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        for ext in _FILE_EXTS:
            input_files.extend(str(f) for f in path.rglob(ext))
    return input_files


def build_artifact_paths(output_files: List[str]) -> Dict[str, str]:
    """Map output-file basenames to their paths for the manifest's artifact index."""
    artifact_paths: Dict[str, str] = {}
    for output_file in output_files[:10]:
        artifact_paths[Path(output_file).name] = output_file
    return artifact_paths


def begin_run(
    service: Any,
    *,
    run_id: str,
    session_id: str,
    input_files: List[str],
) -> None:
    """Begin a run manifest + fingerprint record through the injected service."""
    service.begin_run(
        run_id=run_id,
        session_id=session_id,
        input_files=input_files,
        pipeline_version="1.0",
        rule_version="1.0",
        model_version="1.0",
    )


def end_run(
    service: Any,
    *,
    run_id: str,
    status: str,
    output_path: str,
    entity_count: int,
    relation_count: int,
    llm_call_count: int,
    errors: List[str],
    warnings: List[str],
    artifact_paths: Dict[str, str],
    classification_metrics: Dict[str, Any],
) -> None:
    """Finish a run manifest through the injected service."""
    service.end_run(
        run_id=run_id,
        status=status,
        output_path=output_path,
        entity_count=entity_count,
        relation_count=relation_count,
        llm_call_count=llm_call_count,
        errors=errors[:10],
        warnings=warnings[:10],
        artifact_paths=artifact_paths,
        classification_metrics=classification_metrics,
    )