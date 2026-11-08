"""Agent tool base abstractions.

Defines the ToolResult dataclass and AgentTool Protocol used throughout the
U4 agent layer. All concrete tools (ontology_tools, system_tools, ...) must
implement the AgentTool Protocol.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolResult:
    """Unified result container returned by every AgentTool.

    Attributes:
        success:   Whether the tool completed without errors.
        content:   The main result payload. Can be a dict, str, or any
                   structured object (e.g. FolderAnalysisResult.to_dict()).
        tool_name: Name of the tool that produced this result.
        error:     Human-readable error message when success=False.
        metadata:  Optional additional context (timings, warnings, etc.).
    """

    success: bool
    content: Any
    tool_name: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> str:
        """Convert the result to a plain-text observation for the LLM.

        The loop appends this text as the "Observation:" part of the prompt.
        """
        if not self.success:
            return f"[TOOL ERROR] {self.tool_name}: {self.error}"
        if isinstance(self.content, dict):
            try:
                return json.dumps(self.content, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                return str(self.content)
        return str(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "content": self.content,
            "error": self.error,
            "metadata": self.metadata,
        }


@runtime_checkable
class AgentTool(Protocol):
    """Protocol that every agent tool must satisfy.

    Tools are discovered by the AgentLoop at construction time. Each tool
    exposes a name, a human-readable description (used in the system prompt),
    and a JSON-schema ``parameters`` dict (mirrors OpenAI function-calling
    schema) that tells the LLM which arguments are available.
    """

    #: Short snake_case identifier, e.g. "analyze_folder"
    name: str

    #: Rich description used verbatim in the system prompt
    description: str

    #: JSON Schema object describing the tool's keyword arguments
    parameters: dict[str, Any]

    async def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a ToolResult."""
        ...
