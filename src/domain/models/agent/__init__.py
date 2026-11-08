"""Agent domain models (re-exported from the authoritative source).

Authoritative source for shared agent contracts is
``src.agent.orchestration._protocols``. We re-export the symbols here so
``from src.domain.models.agent import AgentResult, SessionContext`` keeps
working (import path used by tests and legacy code).
"""
from __future__ import annotations

from src.agent.orchestration._protocols import AgentResult, SessionContext

__all__ = [
    "AgentResult",
    "SessionContext",
]
