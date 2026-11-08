"""Shim for replay_pipeline_session.py - BATCH 2 SHIM

This file is a compatibility shim for tests that still import from the old path.
All functions are delegated to canonical implementations.
"""

from __future__ import annotations

from typing import Any, Callable


def replay_pipeline_session(
    *,
    session_id: str,
    summary: bool,
    stage: str | None = None,
    event: str | None = None,
    failures_only: bool = False,
    load_session: Callable[[], dict[str, Any]] | None = None,
    replay_session: Callable[..., dict[str, Any]] | None = None,
    summarize_session: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay pipeline session in the requested mode."""
    if summary:
        return summarize_session(stage=stage, event=event, failures_only=failures_only)
    if stage is not None or event is not None:
        return replay_session(stage=stage, event=event, failures_only=failures_only)
    return load_session()


def build_replay_sse_event_name(*, summary: bool) -> str:
    """Build SSE event name based on replay mode."""
    return "session_replay.summary" if summary else "session_replay.event"


def build_replay_output_metadata(
    *,
    payload: dict[str, Any],
    session_id: str,
    summary: bool,
) -> dict[str, Any]:
    """Build replay output metadata with shared fields."""
    return {
        "mode": "summary" if summary else "events",
        "session_id": session_id,
        "run_id": payload.get("run_id"),
        "request_id": payload.get("request_id"),
        "status": payload.get("status"),
    }


def build_replay_summary_sse_payload(
    payload: dict[str, Any],
    *,
    session_id: str,
    stage: str | None = None,
    event: str | None = None,
    failures_only: bool = False,
) -> dict[str, Any]:
    """Build summary SSE payload."""
    metadata = build_replay_output_metadata(
        payload=payload,
        session_id=session_id,
        summary=True,
    )
    return {
        "session_id": session_id,
        "run_id": payload.get("run_id"),
        "request_id": payload.get("request_id"),
        "status": payload.get("status"),
        "terminal_reason": payload.get("terminal_reason"),
        "mode": "summary",
        "event": "session_replay.summary",
        "filters": {
            "stage": stage,
            "event": event,
            "failures_only": failures_only,
        },
    }


from src.application.use_cases.exec_envelope_use_case import build_replay_exec_envelope
