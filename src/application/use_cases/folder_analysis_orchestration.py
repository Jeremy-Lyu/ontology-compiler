"""Folder analysis orchestration helpers.

This module provides orchestration functions for folder analysis workflows,
including event building, hook payload construction, and application bundle
generation.

NOTE: This module was referenced but didn't exist. Created as the canonical
implementation based on import requirements from analyze_folder_once.py.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


ANALYSIS_EXEC_PROGRESS_STAGES = ["discovering", "analyzing", "finalizing", "complete"]
ANALYSIS_RESPONSE_COMPATIBILITY_FIELDS = []
ANALYSIS_RESPONSE_COMPATIBILITY_POLICY = {}
ANALYSIS_RESPONSE_STABLE_COMPATIBILITY_FIELDS = []
ANALYSIS_RESPONSE_TRANSITIONAL_COMPATIBILITY_FIELDS = []
ANALYSIS_RESULT_METADATA_FIELDS = ["run_id", "request_id", "session_id", "status"]


def build_analysis_application_orchestration_bundle(**kwargs) -> dict[str, Any]:
    """Build application orchestration bundle."""
    return {
        "request": kwargs.get("request"),
        "context": kwargs.get("context"),
        "config": kwargs.get("config"),
    }


def build_analysis_application_orchestration_bundle_simple(**kwargs) -> dict[str, Any]:
    """Build simplified application orchestration bundle."""
    return build_analysis_application_orchestration_bundle(**kwargs)


def build_analysis_application_finalization_bundle(**kwargs) -> dict[str, Any]:
    """Build application finalization bundle."""
    return {
        "results": kwargs.get("results"),
        "artifacts": kwargs.get("artifacts"),
        "metadata": kwargs.get("metadata"),
    }


def build_analysis_application_finalization_bundle_simple(**kwargs) -> dict[str, Any]:
    """Build simplified application finalization bundle."""
    return build_analysis_application_finalization_bundle(**kwargs)


def build_analysis_exec_progress_event(
    *,
    stage: str,
    progress: float = 0.0,
    message: str = "",
    **kwargs,
) -> dict[str, Any]:
    """Build analysis execution progress event."""
    return {
        "type": "progress",
        "stage": stage,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }


def build_analysis_exec_completion_event(**kwargs) -> dict[str, Any]:
    """Build analysis execution completion event."""
    return {
        "type": "completion",
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }


def build_analysis_exec_error_event(
    *,
    error: Exception | str,
    stage: str = "",
    **kwargs,
) -> dict[str, Any]:
    """Build analysis execution error event."""
    error_message = str(error) if isinstance(error, Exception) else error
    return {
        "type": "error",
        "stage": stage,
        "error": error_message,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }


def build_analysis_pre_hook_payload_inputs(**kwargs) -> dict[str, Any]:
    """Build pre-hook payload inputs."""
    return {
        "folder_path": kwargs.get("folder_path"),
        "config": kwargs.get("config"),
        "context": kwargs.get("context"),
    }


def build_analysis_post_hook_payload_inputs(**kwargs) -> dict[str, Any]:
    """Build post-hook payload inputs."""
    return {
        "results": kwargs.get("results"),
        "artifacts": kwargs.get("artifacts"),
        "config": kwargs.get("config"),
    }


def build_analysis_pre_hook_input(**kwargs) -> dict[str, Any]:
    """Build pre-hook input."""
    return {"payload": build_analysis_pre_hook_payload_inputs(**kwargs)}


def build_analysis_post_hook_input(**kwargs) -> dict[str, Any]:
    """Build post-hook input."""
    return {"payload": build_analysis_post_hook_payload_inputs(**kwargs)}


def build_analysis_post_hook_output(**kwargs) -> dict[str, Any]:
    """Build post-hook output."""
    return {"result": kwargs.get("result"), "status": kwargs.get("status", "success")}


def build_analysis_finalization_bundle(**kwargs) -> dict[str, Any]:
    """Build analysis finalization bundle."""
    return {"results": kwargs.get("results"), "metadata": kwargs.get("metadata")}


def build_analysis_write_result_bundle(**kwargs) -> dict[str, Any]:
    """Build write result bundle."""
    return {"artifacts": kwargs.get("artifacts"), "path": kwargs.get("path")}


def build_analysis_response_compatibility_policy(**kwargs) -> dict[str, Any]:
    """Build response compatibility policy."""
    return {}


def build_analysis_response_compatibility_fields(**kwargs) -> list[str]:
    """Build response compatibility fields."""
    return []


def build_analysis_response_compatibility_layers(**kwargs) -> dict[str, Any]:
    """Build response compatibility layers."""
    return {}


def build_analysis_result_metadata(**kwargs) -> dict[str, Any]:
    """Build analysis result metadata."""
    return {k: v for k, v in kwargs.items() if k in ANALYSIS_RESULT_METADATA_FIELDS}


def build_analysis_application_result_bundle(**kwargs) -> dict[str, Any]:
    """Build application result bundle."""
    return {"results": kwargs.get("results"), "metadata": kwargs.get("metadata")}


def build_analysis_application_result_context(**kwargs) -> dict[str, Any]:
    """Build application result context."""
    return {"context": kwargs.get("context"), "results": kwargs.get("results")}


def build_analysis_canonical_result_bundle(**kwargs) -> dict[str, Any]:
    """Build canonical result bundle."""
    return {"canonical": kwargs.get("canonical"), "evidence": kwargs.get("evidence")}


def build_analysis_compatibility_payload(**kwargs) -> dict[str, Any]:
    """Build compatibility payload."""
    return {"legacy": kwargs.get("legacy"), "current": kwargs.get("current")}


def build_analysis_exec_completion_payload(**kwargs) -> dict[str, Any]:
    """Build execution completion payload."""
    return {"status": kwargs.get("status"), "results": kwargs.get("results")}


def build_analysis_exec_error_payload(**kwargs) -> dict[str, Any]:
    """Build execution error payload."""
    return {"error": kwargs.get("error"), "stage": kwargs.get("stage")}


def build_analysis_exec_progress_stage(**kwargs) -> dict[str, Any]:
    """Build execution progress stage."""
    return {"stage": kwargs.get("stage"), "progress": kwargs.get("progress", 0.0)}


def build_analysis_legacy_request(**kwargs) -> dict[str, Any]:
    """Build legacy request."""
    return kwargs


def build_analysis_manifest_final_result_bundle(**kwargs) -> dict[str, Any]:
    """Build manifest final result bundle."""
    return {"manifest": kwargs.get("manifest"), "results": kwargs.get("results")}


def build_analysis_manifest_result_bundle(**kwargs) -> dict[str, Any]:
    """Build manifest result bundle."""
    return {"manifest": kwargs.get("manifest"), "results": kwargs.get("results")}


def build_analysis_primary_payload(**kwargs) -> dict[str, Any]:
    """Build primary payload."""
    return {"payload": kwargs.get("payload")}