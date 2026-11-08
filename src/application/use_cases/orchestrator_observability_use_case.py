"""Canonical observability helpers for pipeline orchestration.

[BATCH 2 + BATCH 4 CONSOLIDATION (2026-06-29)]:
- This module is the **canonical location** for orchestrator observability utilities
- Consolidated from: orchestrator_helpers_use_case.py (deleted), orchestrator_logging_use_case.py (deleted)
- Integrates with: replay_use_cases.py (OrchestratorReplayHandler)

Features:
- ReplayOutputAssembler: Assemble replay outputs for JSON/exec envelope/SSE delivery
- Session recording helpers: record_stage_dispatch_failed, record_unknown_stage, etc.
- Pipeline summary helpers: load_recent_pipeline_summaries, build_pipeline_summary_paths
- Run directory helpers: infer_latest_run_dir
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Generator

from src.application.use_cases.exec_envelope_use_case import build_replay_exec_envelope


class ReplayOutputAssembler:
    """Assemble replay outputs for JSON, exec envelope, or SSE delivery."""

    def __init__(self, *, iter_session_replay_sse: Callable[..., Any]) -> None:
        self._iter_session_replay_sse = iter_session_replay_sse

    def build_json_payload(
        self,
        payload: dict[str, Any],
        *,
        session_id: str,
        stage: str | None,
        event: str | None,
        failures_only: bool,
        summary: bool,
        exec_envelope: bool,
    ) -> dict[str, Any]:
        if not exec_envelope:
            return payload
        return build_replay_exec_envelope(
            payload,
            session_id=session_id,
            stage=stage,
            event=event,
            failures_only=failures_only,
            summary=summary,
        )

    def build_sse_chunks(
        self,
        payload: dict[str, Any],
        *,
        session_id: str,
        stage: str | None,
        event: str | None,
        failures_only: bool,
        summary: bool,
    ) -> Generator[Any, None, None]:
        return self._iter_session_replay_sse(
            payload,
            session_id=session_id,
            summary=summary,
            stage=stage,
            event=event,
            failures_only=failures_only,
        )


def record_stage_dispatch_failed(
    session: Any,
    *,
    stage: str,
    error_type: str,
    detail: str,
) -> None:
    session.record_stage_dispatch_failed(stage=stage, error_type=error_type, detail=detail)


def record_unknown_stage(
    session: Any,
    *,
    stage: str,
    error_type: str,
) -> None:
    session.record_unknown_stage(stage=stage, error_type=error_type)


def record_dry_run(
    session: Any,
    *,
    selected_stages: list[str],
) -> None:
    session.record_dry_run(selected_stages=selected_stages)


def record_hook_denied(
    session: Any,
    *,
    stage: str,
    hook_event: str,
    error_type: str,
    reason: str,
) -> None:
    session.record_hook_denied(
        stage=stage,
        hook_event=hook_event,
        error_type=error_type,
        reason=reason,
    )


def record_stage_result(
    session: Any,
    *,
    result: Any,
    dispatch: dict[str, Any] | None,
) -> None:
    session.record_stage_result(result=result, dispatch=dispatch)


def finish_session_with_terminal_event(
    session: Any,
    *,
    status: str,
    reason: str,
    stage: str | None = None,
    error_type: str | None = None,
    results_count: int | None = None,
) -> None:
    session.finish(
        status=status,
        reason=reason,
        stage=stage,
        error_type=error_type,
        results_count=results_count,
    )


def load_recent_pipeline_summaries(*, runs_root: Path, max_items: int) -> list[dict[str, Any]]:
    if max_items <= 0 or not runs_root.exists():
        return []
    summary_files = sorted(
        (path for path in runs_root.glob("pipeline-*/PIPELINE_RUN_SUMMARY.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    payloads: list[dict[str, Any]] = []
    for path in summary_files[:max_items]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def build_pipeline_summary_paths(*, artifact_store: Any, run_id: str) -> Any:
    return artifact_store.build_pipeline_summary_paths(run_id=run_id)


def write_pipeline_summary_artifacts(
    summary: dict[str, Any],
    *,
    artifact_store: Any,
    paths: Any,
    legacy_warning_label: str,
) -> None:
    artifact_store.write_pipeline_summary(
        summary,
        paths=paths,
        legacy_warning_label=legacy_warning_label,
    )


def infer_latest_run_dir(*, runs_root: Path, run_id_prefix: str | None) -> Path | None:
    if not runs_root.exists():
        return None
    pipeline_dirs = sorted(
        (p for p in runs_root.glob("pipeline-*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not pipeline_dirs:
        return None
    if run_id_prefix:
        filtered = [p for p in pipeline_dirs if p.name.startswith(f"pipeline-{run_id_prefix}")]
        if filtered:
            return filtered[0]
    return pipeline_dirs[0]


__all__ = [
    "ReplayOutputAssembler",
    "record_stage_dispatch_failed",
    "record_unknown_stage",
    "record_dry_run",
    "record_hook_denied",
    "record_stage_result",
    "finish_session_with_terminal_event",
    "load_recent_pipeline_summaries",
    "build_pipeline_summary_paths",
    "write_pipeline_summary_artifacts",
    "infer_latest_run_dir",
]
