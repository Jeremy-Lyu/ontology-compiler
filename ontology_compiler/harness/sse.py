"""SSE (Server-Sent Events) utilities for session replay."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator


def format_sse_event(data: Any, event: str | None = None, event_id: str | None = None) -> str:
    """Format data as SSE event."""
    lines = []
    
    if event_id:
        lines.append(f"id: {event_id}")
    
    if event:
        lines.append(f"event: {event}")
    
    if isinstance(data, str):
        for line in data.split("\n"):
            lines.append(f"data: {line}")
    else:
        lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    
    return "\n".join(lines) + "\n\n"


def iter_session_replay_sse(
    payload: dict[str, Any],
    *,
    session_id: str,
    summary: bool,
    stage: str | None = None,
    event: str | None = None,
    failures_only: bool = False,
) -> Iterator[str]:
    """Iterate session replay events as SSE frames."""
    if summary:
        filtered_payload = {
            "session_id": payload.get("session_id", session_id),
            "status": payload.get("status"),
            "terminal_reason": payload.get("terminal_reason"),
            "mode": "summary",
            "run_id": None,
            "request_id": None,
            "event": "session_replay.summary",
            "filters": {
                "stage": stage,
                "event": event,
                "failures_only": failures_only,
            },
        }
        yield format_sse_event(filtered_payload, "session_replay.summary", f"{session_id}:summary")
    else:
        events = payload.get("events", [])
        for idx, evt in enumerate(events):
            if failures_only and evt.get("status") != "FAILED":
                continue
            if stage and evt.get("stage") != stage:
                continue
            
            evt_with_session = dict(evt)
            evt_with_session["session_id"] = session_id
            evt_with_session["filters"] = {
                "stage": stage,
                "event": event,
                "failures_only": failures_only,
            }
            yield format_sse_event(evt_with_session, "session_replay.event", f"{session_id}:{idx}")


def iter_sse_events(events: list[Dict[str, Any]]) -> Iterator[str]:
    """Iterate SSE formatted events."""
    for event in events:
        evt_data = event.get("data", {})
        evt_event = event.get("event")
        evt_id = event.get("id")
        yield format_sse_event(evt_data, evt_event, evt_id)