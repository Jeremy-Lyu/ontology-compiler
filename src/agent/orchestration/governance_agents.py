"""Governance Agents."""
from __future__ import annotations

from typing import Any, List

from ._protocols import BaseAgent, AgentRole, AgentTask, AgentResult, TaskStatus


class GovernanceAgentBase(BaseAgent):
    """Governance agent base class."""
    
    def __init__(self, agent_id: str, role: AgentRole, name: str, description: str):
        super().__init__(
            agent_id=agent_id,
            role=role,
            name=name,
            description=description,
            capabilities=["scan", "analysis", "reporting"],
        )
    
    def _format_scan_result(
        self,
        phenomenon: str,
        risk: str,
        cause: str,
        files: List[str],
        suggestion: str,
        priority: str = "medium",
        depends_on: List[str] = None,
    ) -> dict[str, Any]:
        """Format scan result."""
        return {
            "phenomenon": phenomenon,
            "risk": risk,
            "cause": cause,
            "files": files,
            "suggestion": suggestion,
            "priority": priority,
            "depends_on": depends_on or [],
        }


class FileParserAgent(GovernanceAgentBase):
    """File parser governance agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="file_parser",
            role=AgentRole(name="FileParser", description="File parsing standardization"),
            name="FileParserAgent",
            description="File parsing standardization scan",
        )


class ConflictGovernorAgent(GovernanceAgentBase):
    """Conflict governor agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="conflict_governor",
            role=AgentRole(name="ConflictGovernor", description="Conflict governance"),
            name="ConflictGovernorAgent",
            description="Conflict governance scan",
        )


class RelationTypeGovernorAgent(GovernanceAgentBase):
    """Relation type governor agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="relation_type_governor",
            role=AgentRole(name="RelationTypeGovernor", description="Relation type governance"),
            name="RelationTypeGovernorAgent",
            description="Relation type governance scan",
        )


class DataContinuityAgent(GovernanceAgentBase):
    """Data continuity agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="data_continuity",
            role=AgentRole(name="DataContinuity", description="Data continuity"),
            name="DataContinuityAgent",
            description="Data continuity scan",
        )


class AssetRagAgent(GovernanceAgentBase):
    """Asset RAG agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="asset_rag",
            role=AgentRole(name="AssetRAG", description="Asset/RAG"),
            name="AssetRagAgent",
            description="Asset/RAG scan",
        )


class ContractAlignerAgent(GovernanceAgentBase):
    """Contract aligner agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="contract_aligner",
            role=AgentRole(name="ContractAligner", description="Contract alignment"),
            name="ContractAlignerAgent",
            description="Contract alignment scan",
        )