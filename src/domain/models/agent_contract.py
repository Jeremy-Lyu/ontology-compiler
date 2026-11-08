"""Agent contract models.

These contracts make the T3 multi-agent role definitions explicit in code so
routing/orchestration layers can rely on stable, versioned metadata.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRoleType(str, Enum):
    ROUTER = "router"
    EXTRACTOR = "extractor"
    VALIDATOR = "validator"
    REPAIR = "repair"


class AgentContract(BaseModel):
    agent_name: str = Field(..., description="Stable agent identifier")
    role: AgentRoleType = Field(..., description="High-level role")
    version: str = Field("v1", description="Contract version")
    objective: str = Field(..., description="Primary objective")
    inputs: List[str] = Field(default_factory=list, description="Expected input fields")
    outputs: List[str] = Field(default_factory=list, description="Expected output fields")
    allowed_skills: List[str] = Field(default_factory=list, description="Skills this agent may call")
    forbidden_actions: List[str] = Field(default_factory=list, description="Actions this agent must not do")
    decision_scope: List[str] = Field(default_factory=list, description="Decision scope of the agent")
    trace_format: str = Field("json+md", description="Preferred trace format")
    failure_policy: str = Field("conservative", description="Failure handling policy")
    fallback_policy: str = Field("fallback-to-minimal", description="Fallback handling policy")
    replayability: bool = Field(True, description="Whether the agent run should be replayable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


__all__ = ["AgentRoleType", "AgentContract"]
