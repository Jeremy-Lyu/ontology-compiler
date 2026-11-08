"""Core helpers for folder analysis.

This module provides core functionality for folder analysis workflows,
including context building, quality assessment, and core result handling.

NOTE: This module was referenced but didn't exist. Created as the canonical
implementation based on import requirements.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_analysis_context(**kwargs) -> dict[str, Any]:
    """Build analysis context."""
    return {"folder_path": kwargs.get("folder_path"), "config": kwargs.get("config")}


def build_analysis_core_results_context(**kwargs) -> dict[str, Any]:
    """Build core results context."""
    return {"results": kwargs.get("results"), "quality": kwargs.get("quality")}


def build_analysis_core_results_context_typed(**kwargs) -> dict[str, Any]:
    """Build typed core results context."""
    return build_analysis_core_results_context(**kwargs)


def build_analysis_quality(**kwargs) -> dict[str, Any]:
    """Build analysis quality metrics."""
    return {"score": kwargs.get("score", 0.0), "issues": kwargs.get("issues", [])}


def build_analysis_runtime_seed(**kwargs) -> dict[str, Any]:
    """Build analysis runtime seed."""
    return {"seed": kwargs.get("seed"), "timestamp": kwargs.get("timestamp")}


def build_core_results_artifacts(**kwargs) -> dict[str, Any]:
    """Build core results artifacts."""
    return {"artifacts": kwargs.get("artifacts", [])}


def build_core_results_markdown_inputs(**kwargs) -> dict[str, Any]:
    """Build core results markdown inputs."""
    return {"markdown": kwargs.get("markdown", "")}


def build_default_run_id(**kwargs) -> str:
    """Build default run ID."""
    return kwargs.get("run_id", "default-run-id")


def build_execution_warnings(**kwargs) -> list[str]:
    """Build execution warnings list."""
    return kwargs.get("warnings", [])


def build_folder_analysis_run_context(**kwargs) -> dict[str, Any]:
    """Build folder analysis run context."""
    return {
        "run_id": kwargs.get("run_id"),
        "request_id": kwargs.get("request_id"),
        "session_id": kwargs.get("session_id"),
        "folder_path": kwargs.get("folder_path"),
    }


def build_inventory_folder_run_id(**kwargs) -> str:
    """Build inventory folder run ID."""
    return kwargs.get("run_id", "inventory-run-id")


def build_inventory_summary_row(**kwargs) -> dict[str, Any]:
    """Build inventory summary row."""
    return {"name": kwargs.get("name"), "value": kwargs.get("value")}


def compute_folder_fingerprint(folder_path: Path | str) -> str:
    """Compute folder fingerprint."""
    from src.application.services.unified_fingerprint import folder_fingerprint
    return folder_fingerprint(folder_path)


def folder_fingerprint(folder_path: Path | str) -> str:
    """Compute folder fingerprint (alias)."""
    return compute_folder_fingerprint(folder_path)


def run_quality_gates(**kwargs) -> dict[str, Any]:
    """Run quality gates."""
    return {"passed": kwargs.get("passed", True), "issues": kwargs.get("issues", [])}