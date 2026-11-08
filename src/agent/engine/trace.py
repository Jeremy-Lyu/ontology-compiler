"""Agent Trace — structured run observability (U4.5).

Records every step of an Agent run (thought, action, observation, timing,
token counts) and writes the completed trace to a JSONL file for replay and
diagnostics.

Architecture
------------
- ``AgentTraceStep`` : immutable record of one ReAct step.
- ``AgentTrace``      : container for a complete run; tracks session_id, goal,
                        start/end time, and all steps.
- ``AgentTraceStore`` : writes completed traces to disk as JSONL (one JSON
                        object per line, one file per day).  Also allows
                        loading the most recent N traces for inspection.

Usage (in AgentLoop)::

    trace = AgentTrace(session_id="abc", goal="Analyse /data")
    trace.add_step(AgentTraceStep(...))
    ...
    trace.finish(success=True, final_answer="Done.")
    store = AgentTraceStore()
    store.save(trace)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default directory for JSONL trace files
DEFAULT_TRACE_DIR: str = "outputs/runs/agent_traces"

# Maximum chars for observation stored in a trace step.
# Prevents single-step observations from bloating the JSONL file.
_MAX_OBS_CHARS: int = 2000


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentTraceStep:
    """One Thought→Action→Observation cycle within a run.

    Attributes:
        step_number:   1-based index.
        thought:       The agent's reasoning text.
        action_name:   Tool invoked (None for Finish steps).
        action_input:  kwargs passed to the tool.
        observation:   Tool result text (truncated to budget).
        is_finish:     True if this is the terminal step.
        error:         Error message if the step failed.
        started_at:    ISO-8601 timestamp when the step began.
        finished_at:   ISO-8601 timestamp when the step ended.
        duration_ms:   Wall-clock duration of the step in milliseconds.
    """

    step_number: int
    thought: str
    action_name: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    is_finish: bool = False
    error: str | None = None
    started_at: str = field(default_factory=lambda: _now_iso())
    finished_at: str | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTrace:
    """Complete record of one Agent run.

    Attributes:
        session_id:   Unique identifier shared across a user conversation.
        run_id:       Unique identifier for this specific run within a session.
        goal:         The user's original query.
        steps:        Ordered list of completed steps.
        started_at:   ISO-8601 timestamp.
        finished_at:  ISO-8601 timestamp (set by ``finish()``).
        success:      Whether the run ended with Finish and no error.
        final_answer: The agent's concluding statement.
    """

    session_id: str
    run_id: str
    goal: str
    steps: list[AgentTraceStep] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: _now_iso())
    finished_at: str | None = None
    success: bool = False
    final_answer: str = ""

    # ------------------------------------------------------------------ #
    # Mutation helpers
    # ------------------------------------------------------------------ #

    def add_step(self, step: AgentTraceStep) -> None:
        """Append a completed step, applying observation budget truncation."""
        if step.observation and len(step.observation) > _MAX_OBS_CHARS:
            step.observation = step.observation[:_MAX_OBS_CHARS] + "…[truncated]"
        self.steps.append(step)

    def finish(self, *, success: bool, final_answer: str) -> None:
        """Seal the trace after the loop ends."""
        self.success = success
        self.final_answer = final_answer
        self.finished_at = _now_iso()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "goal": self.goal,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success": self.success,
            "final_answer": self.final_answer,
            "total_steps": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentTrace":
        steps_raw = data.pop("steps", [])
        data.pop("total_steps", None)  # derived field
        steps = [AgentTraceStep(**s) for s in steps_raw]
        trace = cls(**data)
        trace.steps = steps
        return trace


# ──────────────────────────────────────────────────────────────────────────────
# Trace store
# ──────────────────────────────────────────────────────────────────────────────


class AgentTraceStore:
    """Persists and loads Agent traces as JSONL files.

    One JSONL file is created per calendar day (UTC):
    ``outputs/agent_traces/traces_YYYY-MM-DD.jsonl``

    Each line is a single JSON object (one trace per line).

    Args:
        trace_dir: Directory to store JSONL files. Created on first write.
    """

    def __init__(self, trace_dir: str = DEFAULT_TRACE_DIR) -> None:
        self._trace_dir = Path(trace_dir)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def save(self, trace: AgentTrace) -> Path:
        """Append *trace* to today's JSONL file.  Returns the file path."""
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        path = self._daily_path()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(trace.to_json() + "\n")
        logger.info(
            "AgentTraceStore.save: session=%s run=%s → %s",
            trace.session_id, trace.run_id, path,
        )
        return path

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def load_recent(self, limit: int = 50) -> list[AgentTrace]:
        """Return the *limit* most-recent traces across all JSONL files."""
        if not self._trace_dir.exists():
            return []
        files = sorted(self._trace_dir.glob("traces_*.jsonl"), reverse=True)
        traces: list[AgentTrace] = []
        for f in files:
            traces.extend(self._load_file(f))
            if len(traces) >= limit:
                break
        return traces[:limit]

    def load_by_session(self, session_id: str) -> list[AgentTrace]:
        """Return all traces for a given *session_id*."""
        if not self._trace_dir.exists():
            return []
        files = sorted(self._trace_dir.glob("traces_*.jsonl"), reverse=True)
        matched: list[AgentTrace] = []
        for f in files:
            for t in self._load_file(f):
                if t.session_id == session_id:
                    matched.append(t)
        return matched

    def load_by_run(self, run_id: str) -> AgentTrace | None:
        """Return the trace for a given *run_id*, or None."""
        if not self._trace_dir.exists():
            return None
        for f in sorted(self._trace_dir.glob("traces_*.jsonl"), reverse=True):
            for t in self._load_file(f):
                if t.run_id == run_id:
                    return t
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _daily_path(self) -> Path:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return self._trace_dir / f"traces_{today}.jsonl"

    @staticmethod
    def _load_file(path: Path) -> list[AgentTrace]:
        traces: list[AgentTrace] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        traces.append(AgentTrace.from_dict(data))
                    except Exception as exc:
                        logger.warning("AgentTraceStore: skip malformed line: %s", exc)
        except OSError as exc:
            logger.warning("AgentTraceStore: could not read %s: %s", path, exc)
        return traces


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
