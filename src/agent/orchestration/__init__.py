"""Agent orchestration package.

Exports contract-aware orchestration primitives.

P0-6 (2026-09-09): the zero-consumer agent experiment modules
(orchestrator / skill_registry / skill_router / skill_examples) were retired
to archived/retired_20260909/src_agent_orchestration/; only the shared
_protocols (re-exported by src.domain.models.agent) and governance_agents
remain live.
"""

from ._protocols import AgentRole, AgentTask, AgentResult, TaskStatus

__all__ = [
    "AgentRole",
    "AgentTask",
    "AgentResult",
    "TaskStatus",
]
