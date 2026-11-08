"""Protocol/contract dataclasses for agent orchestration (single source of truth).

Formerly scattered across agent_protocol.py, session_context.py, agent_base.py,
agent_trace_keys.py. The original files remain as thin re-export compatibility
shims so existing import paths stay valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRole:
    name: str
    description: str


@dataclass
class AgentTask:
    task_id: str
    description: str
    status: str = "pending"
    agent_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    task_id: str
    status: str
    output: Any = None
    agent_name: str = ""
    run_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionContext:
    session_id: str
    request_id: str | None = None
    metadata: dict[str, Any] = None
    created_at: str | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BaseAgent:
    agent_id: str
    role: Any
    name: str
    description: str
    capabilities: list[str] = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


@dataclass(frozen=True)
class OrchestrationTraceKey:
    session_id: str = ""
    run_id: str = ""
    task_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
        }


__all__ = [
    "AgentRole", "AgentTask", "AgentResult", "TaskStatus",
    "SessionContext", "BaseAgent", "OrchestrationTraceKey",
]
