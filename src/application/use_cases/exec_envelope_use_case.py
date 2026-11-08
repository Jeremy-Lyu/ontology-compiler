"""Exec envelope use case: build exec envelope payloads for orchestrator output.

Extracted from scripts/run_pipeline_orchestrator.py (previously inlined).
Re-extraction part of P0-ARCH-03 phased refactoring (Plan B Stage 2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_discovery_error_payload(
    *, source: str, errors: list[dict[str, str]]
) -> dict[str, Any]:
    """Build discovery error payload for exec envelope."""
    normalized: list[dict[str, str]] = []
    for item in errors:
        raw_path = item.get("path") or ""
        posix_path = Path(raw_path).as_posix() if raw_path else ""
        error_value = item.get("error", "")
        error_str = str(error_value) if error_value else ""
        kind_value = item.get("kind")
        kind_str = kind_value if kind_value else "manifest"
        normalized.append({
            "kind": kind_str,
            "source": source,
            "path": posix_path,
            "error": error_str,
        })
    return {
        "source": source,
        "count": len(errors),
        "errors": normalized,
    }


def build_registry_listing_payload(
    *,
    items_key: str,
    items: list[dict[str, Any]],
    source: str,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    """Build registry listing payload for exec envelope."""
    payload: dict[str, Any] = {
        items_key: items,
        "discovery_errors": build_discovery_error_payload(source=source, errors=errors),
    }
    return payload


def build_exec_success(kind: str, payload: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Build exec success envelope."""
    envelope: dict[str, Any] = {
        "ok": True,
        "kind": kind,
        "payload": payload,
    }
    envelope.update(extra)
    return envelope


def build_exec_error(kind: str, error: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Build exec error envelope."""
    envelope: dict[str, Any] = {
        "ok": False,
        "kind": kind,
        "error": error,
    }
    envelope.update(extra)
    return envelope


def build_replay_exec_envelope(
    payload: dict[str, Any],
    *,
    session_id: str,
    stage: str | None = None,
    event: str | None = None,
    failures_only: bool = False,
    summary: bool = False,
) -> dict[str, Any]:
    """Build replay exec envelope."""
    envelope: dict[str, Any] = {
        "ok": True,
        "kind": "session_replay",
        "source": "orchestrator",
        "phase": "session_replay",
        "mode": "summary" if summary else "events",
        "session_id": session_id,
        "run_id": payload.get("run_id"),
        "request_id": payload.get("request_id"),
        "status": payload.get("status"),
        "payload": payload,
        "filters": {
            "stage": stage,
            "event": event,
            "failures_only": failures_only,
        },
    }
    return envelope


def build_dry_run_payload(
    *,
    profile: str,
    stages: list[str],
    session_id: str,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    """Build dry run payload."""
    return {
        "profile": profile,
        "stages": stages,
        "session_id": session_id,
        "bootstrap": bootstrap,
    }


def build_pipeline_run_payload(
    summary: dict[str, Any],
    *,
    summary_path: str | Path,
    latest_path: str | Path,
) -> dict[str, Any]:
    """Build pipeline run payload."""
    def _path_str(value: Any) -> str:
        return value.as_posix() if isinstance(value, Path) else str(value)

    return {
        "run_id": summary.get("run_id", ""),
        "request_id": summary.get("request_id", ""),
        "session_id": summary.get("session_id", ""),
        "overall_status": summary.get("overall_status", ""),
        "results": summary.get("results", []),
        "summary": summary,
        "summary_path": _path_str(summary_path),
        "latest_path": _path_str(latest_path),
    }
