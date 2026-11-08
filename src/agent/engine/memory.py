"""Agent working memory (session-scoped).

Maintains conversation history and a key-value working context for the
duration of one agent run. Intentionally kept separate from the long-term
Capsule experience store (which belongs to the Application layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    """A single turn in the agent conversation.

    role values mirror OpenAI chat roles:
        "system"    — system prompt (injected once at start)
        "user"      — the human goal / follow-up instruction
        "assistant" — the agent's THOUGHT + ACTION text
        "tool"      — the Observation returned after a tool call
    """

    role: str
    content: str
    tool_name: str | None = None  # populated for role=="tool"


@dataclass
class AgentMemory:
    """Session-scoped working memory for one agent run.

    Usage::

        memory = AgentMemory(goal="Analyze /data/TBH/Cycle Stock")
        memory.add_thought("I should call analyze_folder first.")
        memory.add_observation(obs_text, tool_name="analyze_folder")
        messages = memory.to_llm_messages()
    """

    goal: str
    messages: list[AgentMessage] = field(default_factory=list)
    working_context: dict[str, Any] = field(default_factory=dict)
    step_count: int = 0

    # ------------------------------------------------------------------ #
    # Mutation helpers
    # ------------------------------------------------------------------ #

    def add_system(self, content: str) -> None:
        self.messages.append(AgentMessage(role="system", content=content))

    def add_user(self, content: str) -> None:
        self.messages.append(AgentMessage(role="user", content=content))

    def add_thought(self, content: str) -> None:
        """Record the agent's thought + action declaration."""
        self.messages.append(AgentMessage(role="assistant", content=content))
        self.step_count += 1

    def add_observation(self, content: str, tool_name: str) -> None:
        """Record the tool's observation."""
        self.messages.append(
            AgentMessage(role="tool", content=content, tool_name=tool_name)
        )

    def add_finish(self, content: str) -> None:
        """Record the agent's final answer."""
        self.messages.append(AgentMessage(role="assistant", content=content))
        self.step_count += 1

    def set_context(self, key: str, value: Any) -> None:
        self.working_context[key] = value

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_llm_messages(self) -> list[dict[str, str]]:
        """Format memory as a flat list of dicts for LLM prompt building."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_prompt_text(self) -> str:
        """Format memory as a single text block for simple string-based LLMs."""
        lines: list[str] = []
        for msg in self.messages:
            role_label = {
                "system": "System",
                "user": "User",
                "assistant": "Assistant",
                "tool": f"Observation [{msg.tool_name}]",
            }.get(msg.role, msg.role.capitalize())
            lines.append(f"### {role_label}\n{msg.content}")
        return "\n\n".join(lines)
