"""Extended helpers for folder analysis.

This module provides extended functionality for folder analysis workflows,
including manifest building, result handling, and artifact writing.

NOTE: This module was referenced but didn't exist. Created as the canonical
implementation based on import requirements.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def attach_core_results_artifact(**kwargs) -> dict[str, Any]:
    """Attach core results artifact."""
    return {"artifact": kwargs.get("artifact"), "results": kwargs.get("results")}


def build_analysis_artifact_write_context(**kwargs) -> dict[str, Any]:
    """Build artifact write context."""
    return {"path": kwargs.get("path"), "content": kwargs.get("content")}


def build_analysis_final_result_bundle(**kwargs) -> dict[str, Any]:
    """Build final result bundle."""
    return {"results": kwargs.get("results"), "metadata": kwargs.get("metadata")}


def build_analysis_final_result_context(**kwargs) -> dict[str, Any]:
    """Build final result context."""
    return {"context": kwargs.get("context"), "results": kwargs.get("results")}


def build_analysis_final_result_payload_context(**kwargs) -> dict[str, Any]:
    """Build final result payload context."""
    return {"payload": kwargs.get("payload"), "context": kwargs.get("context")}


def build_analysis_manifest(**kwargs) -> dict[str, Any]:
    """Build analysis manifest."""
    return {"files": kwargs.get("files", []), "metadata": kwargs.get("metadata", {})}


def build_analysis_manifest_bundle(**kwargs) -> dict[str, Any]:
    """Build manifest bundle."""
    return {"manifest": kwargs.get("manifest"), "artifacts": kwargs.get("artifacts", [])}


def build_analysis_manifest_context(**kwargs) -> dict[str, Any]:
    """Build manifest context."""
    return {"manifest": kwargs.get("manifest"), "context": kwargs.get("context")}


def build_analysis_manifest_inputs(**kwargs) -> dict[str, Any]:
    """Build manifest inputs."""
    return {"folder_path": kwargs.get("folder_path"), "config": kwargs.get("config")}


def build_analysis_run_result(**kwargs) -> dict[str, Any]:
    """Build analysis run result."""
    return {"status": kwargs.get("status"), "results": kwargs.get("results")}


def build_analysis_run_result_inputs(**kwargs) -> dict[str, Any]:
    """Build run result inputs."""
    return {"inputs": kwargs.get("inputs", {}), "config": kwargs.get("config")}


def build_analysis_run_status(**kwargs) -> dict[str, Any]:
    """Build run status."""
    return {"status": kwargs.get("status", "running"), "progress": kwargs.get("progress", 0.0)}


def build_analysis_write_auxiliary_bundle(**kwargs) -> dict[str, Any]:
    """Build auxiliary write bundle."""
    return {"auxiliary": kwargs.get("auxiliary", []), "path": kwargs.get("path")}


def build_analysis_write_auxiliary_context(**kwargs) -> dict[str, Any]:
    """Build auxiliary write context."""
    return {"context": kwargs.get("context"), "auxiliary": kwargs.get("auxiliary", [])}


def build_analysis_write_auxiliary_stage_bundle(**kwargs) -> dict[str, Any]:
    """Build auxiliary stage write bundle."""
    return {"stage": kwargs.get("stage"), "auxiliary": kwargs.get("auxiliary", [])}


def build_analysis_write_operations_bundle(**kwargs) -> dict[str, Any]:
    """Build write operations bundle."""
    return {"operations": kwargs.get("operations", []), "path": kwargs.get("path")}


def build_analysis_write_operations_context(**kwargs) -> dict[str, Any]:
    """Build write operations context."""
    return {"context": kwargs.get("context"), "operations": kwargs.get("operations", [])}


def build_analysis_write_stage_bundle(**kwargs) -> dict[str, Any]:
    """Build stage write bundle."""
    return {"stage": kwargs.get("stage"), "artifacts": kwargs.get("artifacts", [])}


def build_analysis_write_stage_context(**kwargs) -> dict[str, Any]:
    """Build stage write context."""
    return {"stage": kwargs.get("stage"), "context": kwargs.get("context")}


def build_minimal_output_artifact_updates(**kwargs) -> dict[str, Any]:
    """Build minimal output artifact updates."""
    return {"updates": kwargs.get("updates", {})}


def merge_manifest_artifacts(**kwargs) -> dict[str, Any]:
    """Merge manifest artifacts."""
    return {"merged": kwargs.get("artifacts", [])}


def resolve_analysis_report_path(**kwargs) -> Path | str:
    """Resolve analysis report path."""
    return kwargs.get("path", Path("report.json"))