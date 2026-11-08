from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionManager:
    """Lifecycle-oriented wrapper around `SessionStore`."""

    def __init__(self, store: "SessionStore", session_id: str) -> None:
        self.store = store
        self.session_id = session_id

    def append_event(self, event: dict[str, Any]) -> None:
        self.store.append_event(self.session_id, event)

    def record_dry_run(self, *, selected_stages: list[str]) -> None:
        self.append_event(
            {
                "event": "dry_run",
                "selected_stages": selected_stages,
            }
        )

    def record_stage_dispatch_failed(self, *, stage: str, error_type: str, detail: str) -> None:
        self.append_event(
            {
                "event": "stage_dispatch_failed",
                "stage": stage,
                "status": "FAILED",
                "error_type": error_type,
                "detail": detail,
            }
        )

    def record_unknown_stage(self, *, stage: str, error_type: str) -> None:
        self.append_event(
            {
                "event": "unknown_stage",
                "stage": stage,
                "status": "FAILED",
                "error_type": error_type,
            }
        )

    def record_hook_denied(self, *, stage: str, hook_event: str, error_type: str, reason: str | None) -> None:
        self.append_event(
            {
                "event": "hook_denied",
                "stage": stage,
                "hook_event": hook_event,
                "status": "FAILED",
                "error_type": error_type,
                "reason": reason,
            }
        )

    def record_stage_result(self, *, result: Any, dispatch: dict[str, Any] | None = None) -> None:
        event_dispatch = getattr(result, "dispatch", None) or dispatch or {}
        self.append_event(
            {
                "event": "stage_result",
                "stage": result.stage,
                "status": result.status,
                "duration_ms": result.duration_ms,
                "error_type": result.error_type,
                "action_hint": result.action_hint,
                "command": result.command,
                "warnings": result.warnings,
                "metrics": result.metrics,
                "dispatch": event_dispatch,
            }
        )

    def finish(
        self,
        *,
        status: str,
        reason: str,
        stage: str | None = None,
        error_type: str | None = None,
        results_count: int | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "event": "run_finished",
            "status": status,
            "reason": reason,
        }
        if stage is not None:
            event["stage"] = stage
        if error_type is not None:
            event["error_type"] = error_type
        if results_count is not None:
            event["results_count"] = results_count
        self.append_event(event)
        self.store.finish(self.session_id, status=status)

    def load(self) -> dict[str, Any]:
        return self.store.load(self.session_id)

    def replay(
        self,
        *,
        stage: str | None = None,
        event: str | None = None,
        failures_only: bool = False,
    ) -> dict[str, Any]:
        return self.store.replay(
            self.session_id,
            stage=stage,
            event=event,
            failures_only=failures_only,
        )

    def summarize(
        self,
        *,
        stage: str | None = None,
        event: str | None = None,
        failures_only: bool = False,
    ) -> dict[str, Any]:
        return self.store.summarize(
            self.session_id,
            stage=stage,
            event=event,
            failures_only=failures_only,
        )


class SessionStore:
    """Persist/replay pipeline session traces."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path("outputs/ontology/sessions")
        self.root.mkdir(parents=True, exist_ok=True)

    def session_path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def start(self, *, session_id: str, run_id: str, request_id: str, profile: str, bootstrap: dict[str, Any]) -> None:
        payload = {
            "session_id": session_id,
            "run_id": run_id,
            "request_id": request_id,
            "profile": profile,
            "started_at": datetime.now().isoformat(),
            "bootstrap": bootstrap,
            "events": [],
            "status": "RUNNING",
        }
        self._write(session_id, payload)

    def append_event(self, session_id: str, event: dict[str, Any]) -> None:
        payload = self._read(session_id)
        payload.setdefault("events", []).append(
            {
                "ts": datetime.now().isoformat(),
                **event,
            }
        )
        self._write(session_id, payload)

    def finish(self, session_id: str, *, status: str) -> None:
        payload = self._read(session_id)
        payload["finished_at"] = datetime.now().isoformat()
        payload["status"] = status
        self._write(session_id, payload)

    def load(self, session_id: str) -> dict[str, Any]:
        return self._read(session_id)

    def replay(
        self,
        session_id: str,
        *,
        stage: str | None = None,
        event: str | None = None,
        failures_only: bool = False,
    ) -> dict[str, Any]:
        payload = self._read(session_id)
        events = list(payload.get("events") or [])

        if stage:
            events = [item for item in events if item.get("stage") == stage]
        if event:
            events = [item for item in events if item.get("event") == event]
        if failures_only:
            events = [item for item in events if self._is_failure_event(item)]

        return {
            **payload,
            "events": events,
        }

    def summarize(
        self,
        session_id: str,
        *,
        stage: str | None = None,
        event: str | None = None,
        failures_only: bool = False,
    ) -> dict[str, Any]:
        payload = self.replay(session_id, stage=stage, event=event, failures_only=failures_only)
        events = list(payload.get("events") or [])
        stage_counts = Counter(str(item.get("stage")) for item in events if item.get("stage"))
        event_counts = Counter(str(item.get("event")) for item in events if item.get("event"))
        failure_events = [item for item in events if self._is_failure_event(item)]
        terminal_event = self._extract_terminal_event(payload.get("events") or [])

        return {
            "session_id": payload.get("session_id"),
            "run_id": payload.get("run_id"),
            "request_id": payload.get("request_id"),
            "profile": payload.get("profile"),
            "status": payload.get("status"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "filters": {
                "stage": stage,
                "event": event,
                "failures_only": failures_only,
            },
            "event_count": len(events),
            "failure_count": len(failure_events),
            "stages": dict(stage_counts),
            "events": dict(event_counts),
            "terminal_reason": terminal_event.get("reason") if terminal_event else None,
            "terminal_stage": terminal_event.get("stage") if terminal_event else None,
            "terminal_error_type": terminal_event.get("error_type") if terminal_event else None,
            "terminal_results_count": terminal_event.get("results_count") if terminal_event else None,
            "failures": failure_events,
        }

    def _extract_terminal_event(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        for item in reversed(events):
            if item.get("event") == "run_finished":
                return item
        return None

    def _is_failure_event(self, event: dict[str, Any]) -> bool:
        status = str(event.get("status") or "").upper()
        if status == "FAILED":
            return True
        if str(event.get("event") or "") == "hook_denied":
            return True
        return bool(event.get("error_type"))

    def _read(self, session_id: str) -> dict[str, Any]:
        path = self.session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"session not found: {session_id}")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid session payload: {path}")
        return payload

    def _write(self, session_id: str, payload: dict[str, Any]) -> None:
        path = self.session_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
