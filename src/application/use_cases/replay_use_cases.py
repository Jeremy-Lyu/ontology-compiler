"""Replay Use Cases - Consolidated session replay logic.

This module consolidates all replay-related use cases:
- ReplayOutputAssembler: Assemble replay outputs for json, exec envelope, or SSE delivery
- OrchestratorReplayHandler: Handler for orchestrator replay requests
- ReplayRequest/ReplayHandlerResult: Data models for replay requests
- replay_pipeline_session: Replay pipeline session in requested mode
- build_replay_sse_event_name: Build SSE event name based on replay mode
- build_replay_output_metadata: Build replay output metadata with shared fields
- build_replay_summary_sse_payload: Build summary SSE payload

Extracted from scripts/run_pipeline_orchestrator.py as part of P0-ARCH-03 phased refactoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generator


@dataclass
class ReplayRequest:
    """Request for orchestrator replay."""
    session_id: str
    stage: str | None
    event: str | None
    failures_only: bool
    summary: bool
    replay_sse: bool
    exec_envelope: bool


@dataclass
class ReplayHandlerResult:
    """Result from replay handler."""
    exit_code: int
    is_sse: bool
    output: Any = None
    sse_chunks: Generator[Any, None, None] | None = None


class ReplayOutputAssembler:
    """Assemble replay outputs for json, exec envelope, or SSE delivery."""

    def __init__(self, *, iter_session_replay_sse: Callable[..., Any]) -> None:
        """Initialise with an SSE iterator callable."""
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
        """Build replay JSON payload, optionally wrapped in exec envelope."""
        if not exec_envelope:
            return payload
        from src.application.use_cases.exec_envelope_use_case import build_replay_exec_envelope

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
        """Build replay SSE chunks for the requested replay mode."""
        return self._iter_session_replay_sse(
            payload,
            session_id=session_id,
            summary=summary,
            stage=stage,
            event=event,
            failures_only=failures_only,
        )


class OrchestratorReplayHandler:
    """Handler for orchestrator replay requests."""

    def __init__(self, *, replay_output_assembler: Any) -> None:
        self._assembler = replay_output_assembler

    def handle(
        self,
        request: ReplayRequest,
        *,
        load_session: Callable[[], dict[str, Any]],
        replay_session: Callable[..., Any],
        summarize_session: Callable[..., Any],
        emit_error: Callable[..., int],
    ) -> ReplayHandlerResult:
        """Handle replay request."""
        try:
            if request.summary:
                payload = summarize_session(stage=request.stage, event=request.event, failures_only=request.failures_only)
            elif request.stage is not None or request.event is not None:
                payload = replay_session(stage=request.stage, event=request.event, failures_only=request.failures_only)
            else:
                payload = load_session()

            if request.replay_sse:
                sse_chunks = self._assembler.build_sse_chunks(
                    payload,
                    session_id=request.session_id,
                    stage=request.stage,
                    event=request.event,
                    failures_only=request.failures_only,
                    summary=request.summary,
                )
                return ReplayHandlerResult(exit_code=0, is_sse=True, sse_chunks=sse_chunks)
            else:
                output = self._assembler.build_json_payload(
                    payload,
                    session_id=request.session_id,
                    stage=request.stage,
                    event=request.event,
                    failures_only=request.failures_only,
                    summary=request.summary,
                    exec_envelope=request.exec_envelope,
                )
                return ReplayHandlerResult(exit_code=0, is_sse=False, output=output)

        except FileNotFoundError as e:
            from src.domain.errors import ConfigError, classify_error
            meta = classify_error(ConfigError(str(e)))
            exit_code = emit_error(
                meta,
                detail=str(e),
                exec_json=request.exec_envelope,
                kind="session_replay_error",
                phase="session_replay",
                run_id=None,
                request_id=None,
                session_id=request.session_id,
                extra_context={
                    "filters": {
                        "stage": request.stage,
                        "event": request.event,
                        "failures_only": request.failures_only,
                        "summary": request.summary,
                    }
                },
            )
            return ReplayHandlerResult(exit_code=exit_code, is_sse=False)

        except Exception as e:
            from src.domain.errors import PipelineError, classify_error
            meta = classify_error(PipelineError(str(e)))
            exit_code = emit_error(
                meta,
                detail=str(e),
                exec_json=request.exec_envelope,
                kind="session_replay_error",
                phase="session_replay",
                run_id=None,
                request_id=None,
                session_id=request.session_id,
                extra_context={
                    "filters": {
                        "stage": request.stage,
                        "event": request.event,
                        "failures_only": request.failures_only,
                        "summary": request.summary,
                    }
                },
            )
            return ReplayHandlerResult(exit_code=exit_code, is_sse=False)


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