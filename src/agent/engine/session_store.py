"""Agent Session Store — AgentMemory persistence & recovery (U4.5).

Allows a conversation to be interrupted and resumed by serializing the
full ``AgentMemory`` (goal + message history + working context) to a
JSON file on disk, keyed by ``session_id``.

Architecture
------------
- One session = one JSON file: ``outputs/agent_sessions/<session_id>.json``
- ``AgentSessionStore.save(session_id, memory)``  — serialize and write
- ``AgentSessionStore.load(session_id)``           — deserialize and return
- ``AgentSessionStore.list_sessions()``            — enumerate stored sessions

The on-disk format mirrors the ``AgentMemory`` public attributes so that
future schema changes (new fields) can be handled gracefully with defaults.

Usage::

    store = AgentSessionStore()

    # Save after each run to preserve conversation context
    store.save(session_id, memory)

    # Restore on the next request with the same session_id
    memory = store.load(session_id)
    if memory is None:
        memory = AgentMemory(goal=user_query)  # brand-new session
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.agent.engine.memory import AgentMemory, AgentMessage

logger = logging.getLogger(__name__)

# Default directory for session JSON files
DEFAULT_SESSION_DIR: str = "outputs/runs/agent_sessions"


class AgentSessionStore:
    """Persist and restore ``AgentMemory`` by session ID.

    Args:
        session_dir: Directory that holds one JSON file per session.
                     Created on first write.
    """

    def __init__(self, session_dir: str = DEFAULT_SESSION_DIR) -> None:
        self._session_dir = Path(session_dir)

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def save(self, session_id: str, memory: AgentMemory) -> Path:
        """Serialize *memory* and write it to ``<session_dir>/<session_id>.json``.

        Returns the path that was written.
        """
        self._session_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(session_id)
        data = self._memory_to_dict(memory)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.info("AgentSessionStore.save: session=%s → %s", session_id, path)
        return path

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def load(self, session_id: str) -> AgentMemory | None:
        """Load and deserialize a session.  Returns *None* if not found."""
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            memory = self._dict_to_memory(data)
            logger.info("AgentSessionStore.load: session=%s (%d messages)", session_id, len(memory.messages))
            return memory
        except Exception as exc:
            logger.warning("AgentSessionStore.load: failed for session=%s: %s", session_id, exc)
            return None

    def exists(self, session_id: str) -> bool:
        """Return True if a persisted session exists for *session_id*."""
        return self._path(session_id).exists()

    # ------------------------------------------------------------------ #
    # List
    # ------------------------------------------------------------------ #

    def list_sessions(self) -> list[str]:
        """Return a sorted list of all stored session IDs."""
        if not self._session_dir.exists():
            return []
        return sorted(
            p.stem for p in self._session_dir.glob("*.json")
        )

    def delete(self, session_id: str) -> bool:
        """Delete a session file.  Returns True if deleted, False if not found."""
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            logger.info("AgentSessionStore.delete: session=%s", session_id)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _memory_to_dict(memory: AgentMemory) -> dict[str, Any]:
        return {
            "goal": memory.goal,
            "step_count": memory.step_count,
            "working_context": memory.working_context,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_name": m.tool_name,
                }
                for m in memory.messages
            ],
        }

    @staticmethod
    def _dict_to_memory(data: dict[str, Any]) -> AgentMemory:
        memory = AgentMemory(goal=data.get("goal", ""))
        memory.step_count = int(data.get("step_count", 0))
        memory.working_context = dict(data.get("working_context", {}))
        for raw in data.get("messages", []):
            memory.messages.append(
                AgentMessage(
                    role=raw.get("role", "user"),
                    content=raw.get("content", ""),
                    tool_name=raw.get("tool_name"),
                )
            )
        return memory

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _path(self, session_id: str) -> Path:
        # Sanitize session_id to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self._session_dir / f"{safe_id}.json"
