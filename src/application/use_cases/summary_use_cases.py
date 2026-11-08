"""Summary Use Cases - Consolidated pipeline summary logic.

This module consolidates all summary-related use cases:
- build_pipeline_summary: Build complete pipeline summary from execution result
- build_pipeline_run_payload: Build pipeline run payload with path references
- build_pipeline_summary_paths: Build pipeline summary paths via artifact store
- write_pipeline_summary_artifacts: Write pipeline summary artifacts via artifact store
- Internal helpers for ASI governance, incremental efficiency, and patch evidence

Extracted from scripts/run_pipeline_orchestrator.py as part of P0-ARCH-03 phased refactoring.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def _extract_patch_evidence_summary(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract patch evidence summary from stage results."""
    for stage in results:
        if str(stage.get("stage") or "") != "guard":
            continue
        metrics = stage.get("metrics") if isinstance(stage, dict) else None
        if not isinstance(metrics, dict):
            continue
        report = metrics.get("patch_dry_run_report")
        op_count = metrics.get("patch_evidence_operation_count")
        result_status = metrics.get("patch_evidence_result_status")
        rollback_ok = metrics.get("patch_evidence_rollback_simulation_ok")
        has_patch_evidence = any(
            value is not None and value != ""
            for value in (report, op_count, result_status, rollback_ok)
        )
        if not has_patch_evidence:
            continue
        return {
            "report_path": report,
            "operation_count": op_count,
            "result_status": result_status,
            "rollback_simulation_ok": rollback_ok,
        }
    return None


def _build_asi_governance_summary(
    *,
    execution_results: list[dict[str, Any]],
    patch_evidence_summary: dict[str, Any] | None,
    recent_pipeline_summaries: list[dict[str, Any]] | None,
    asi_threshold: float,
    asi_low_windows_to_pause: int,
) -> dict[str, Any]:
    """Build ASI governance summary from execution results."""
    status_weights = {"SUCCESS": 1.0, "WARNING": 0.6, "FAILED": 0.0}
    weighted = [status_weights.get(str(item.get("status") or "").upper(), 0.0) for item in execution_results]
    stage_health_score = (sum(weighted) / len(weighted)) if weighted else 0.0
    patch_score = 1.0
    if patch_evidence_summary is not None:
        patch_score = 1.0 if bool(patch_evidence_summary.get("rollback_simulation_ok")) else 0.0
    asi_score = round((0.8 * stage_health_score) + (0.2 * patch_score), 4)
    prior_scores: list[float] = []
    for payload in list(recent_pipeline_summaries or []):
        governance = payload.get("governance") if isinstance(payload, dict) else None
        if isinstance(governance, dict) and governance.get("asi_score") is not None:
            try:
                prior_scores.append(float(governance.get("asi_score")))
                continue
            except (TypeError, ValueError):
                pass
        try:
            prior_scores.append(float(payload.get("asi_score")))
        except (TypeError, ValueError, AttributeError):
            continue
    low_asi_streak = 0
    for score in [asi_score, *prior_scores]:
        if score < asi_threshold:
            low_asi_streak += 1
            continue
        break
    return {
        "asi_score": asi_score,
        "asi_threshold": asi_threshold,
        "asi_low_windows_to_pause": asi_low_windows_to_pause,
        "low_asi_streak": low_asi_streak,
        "pause_required": low_asi_streak >= max(1, asi_low_windows_to_pause),
    }


def _extract_stage_duration_ms(*, results: list[dict[str, Any]], stage_name: str) -> int | None:
    """Extract duration_ms for a given stage from results."""
    for stage in results:
        if str(stage.get("stage") or "") != stage_name:
            continue
        raw = stage.get("duration_ms")
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    return None


def _build_incremental_efficiency_metrics(
    *,
    execution_results: list[dict[str, Any]],
    incremental_context: dict[str, Any],
) -> dict[str, Any]:
    """Compute incremental efficiency metrics."""
    candidate_count = int(incremental_context.get("candidate_count") or 0)
    changed_count = len(incremental_context.get("changed_files") or [])
    unchanged_count = len(incremental_context.get("unchanged_files") or [])
    skip_ratio = round((unchanged_count / candidate_count), 4) if candidate_count > 0 else 0.0
    changed_ratio = round((changed_count / candidate_count), 4) if candidate_count > 0 else 0.0
    catalog_duration_ms = _extract_stage_duration_ms(results=execution_results, stage_name="catalog")
    if catalog_duration_ms is None or changed_count <= 0 or candidate_count <= 0:
        return {
            "skip_ratio": skip_ratio,
            "changed_ratio": changed_ratio,
            "catalog_duration_ms": catalog_duration_ms,
            "estimated_full_catalog_duration_ms": None,
            "estimated_time_saved_ms": None,
            "estimated_time_saved_ratio": None,
        }
    per_processed_file_ms = catalog_duration_ms / changed_count
    estimated_full_catalog_duration_ms = int(round(per_processed_file_ms * candidate_count))
    estimated_time_saved_ms = max(0, estimated_full_catalog_duration_ms - catalog_duration_ms)
    estimated_time_saved_ratio = (
        round((estimated_time_saved_ms / estimated_full_catalog_duration_ms), 4)
        if estimated_full_catalog_duration_ms > 0
        else 0.0
    )
    return {
        "skip_ratio": skip_ratio,
        "changed_ratio": changed_ratio,
        "catalog_duration_ms": catalog_duration_ms,
        "estimated_full_catalog_duration_ms": estimated_full_catalog_duration_ms,
        "estimated_time_saved_ms": estimated_time_saved_ms,
        "estimated_time_saved_ratio": estimated_time_saved_ratio,
    }


def build_pipeline_run_payload(
    summary: dict[str, Any],
    *,
    summary_path: Path | str,
    latest_path: Path | str,
) -> dict[str, Any]:
    """Build pipeline run payload with summary and path references."""
    def _path_str(value: Any) -> str:
        return value.as_posix() if isinstance(value, Path) else str(value)

    summary_refs = {
        "summary_path": _path_str(summary_path),
        "latest_path": _path_str(latest_path),
    }
    return {
        "summary": summary,
        **summary_refs,
        "run_id": summary.get("run_id"),
        "request_id": summary.get("request_id"),
        "session_id": summary.get("session_id"),
        "overall_status": summary.get("overall_status"),
        "results": summary.get("results"),
    }


def build_pipeline_summary(
    execution_result: Any,
    *,
    started_at: str,
    profile: str,
    run_id: str,
    request_id: str,
    session_id: str,
    bootstrap: dict[str, Any],
    stages: list[str],
    incremental_context: dict[str, Any],
    recent_pipeline_summaries: list[dict[str, Any]] | None = None,
    asi_threshold: float = 0.75,
    asi_low_windows_to_pause: int = 3,
) -> dict[str, Any]:
    """Build a complete pipeline summary from execution result."""
    execution_summary = execution_result.to_summary_payload()
    patch_evidence_summary = _extract_patch_evidence_summary(execution_summary["results"])
    incremental_efficiency = _build_incremental_efficiency_metrics(
        execution_results=execution_summary["results"],
        incremental_context=incremental_context,
    )
    governance = _build_asi_governance_summary(
        execution_results=execution_summary["results"],
        patch_evidence_summary=patch_evidence_summary,
        recent_pipeline_summaries=recent_pipeline_summaries,
        asi_threshold=asi_threshold,
        asi_low_windows_to_pause=asi_low_windows_to_pause,
    )
    return {
        "generated_at": datetime.now().isoformat(),
        "started_at": started_at,
        "profile": profile,
        "run_id": run_id,
        "request_id": request_id,
        "session_id": session_id,
        "bootstrap": bootstrap,
        "stages": stages,
        "overall_status": execution_summary["overall_status"],
        "incremental_stats": {
            "mode": incremental_context.get("mode"),
            "candidate_count": incremental_context.get("candidate_count"),
            "changed_count": len(incremental_context.get("changed_files") or []),
            "unchanged_count": len(incremental_context.get("unchanged_files") or []),
            "new_count": len(incremental_context.get("new_files") or []),
            "removed_count": len(incremental_context.get("removed_files") or []),
            "manifest_path": incremental_context.get("manifest_path"),
            "efficiency": incremental_efficiency,
        },
        "patch_evidence_summary": patch_evidence_summary,
        "governance": governance,
        "results": execution_summary["results"],
    }


def build_pipeline_summary_paths(*, artifact_store: Any, run_id: str) -> Any:
    """Build pipeline summary paths via artifact store."""
    return artifact_store.build_pipeline_summary_paths(run_id=run_id)


def write_pipeline_summary_artifacts(
    summary: dict[str, Any],
    *,
    artifact_store: Any,
    paths: Any,
    legacy_warning_label: str,
) -> None:
    """Write pipeline summary artifacts via artifact store."""
    artifact_store.write_pipeline_summary(
        summary,
        paths=paths,
        legacy_warning_label=legacy_warning_label,
    )