"""Observability bounded-context port definitions.

Ports and DTOs for hooks, identity/evidence planes and multi-agent
analysis results. The neutral compiler-facing contracts live in
`core.contracts.ports`; they are re-exported here so application-layer
imports share the canonical class identity. Split from `all_ports.py`
during P2 port classification (§9.8).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from core.contracts.ports import (
    EvidencePlanePort,
    HookResult,
    HookRunnerPort,
    IdentityRegistryPort,
)


@dataclass(frozen=True)
class AnalysisFinding:
    """Single finding from multi-agent analysis."""
    entity_id: str
    finding_type: str  # dependency, gap, contradiction, hypothesis
    description: str
    confidence: float
    source_hypothesis: Optional[str] = None


@dataclass(frozen=True)
class AnalysisResult:
    """Unified result format for multi-agent analysis operations.

    This port decouples the data_analysis subpackage from the main
    compilation pipeline, enabling different analysis implementations.

    Phase 6-ARCH-1 (2026-06-22)
    """
    status: str  # SUCCESS, PARTIAL, LOCAL_ONLY
    question: str
    converged: bool
    total_rounds: int
    findings: list[AnalysisFinding]
    confidence: float
    report_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AnalysisResultPort(ABC):
    """Port interface for multi-agent data analysis results.

    Decouples ontology_compiler.services.data_analysis from the main
    compilation pipeline, enabling:
    - Different analysis implementations (local/LLM)
    - Mock analysis for testing
    - Alternative analysis backends

    Phase 6-ARCH-1 (2026-06-22)
    """

    @abstractmethod
    async def analyze(
        self,
        extraction_result: dict[str, Any],
        question: str,
        background: Optional[str] = None,
        max_rounds: int = 3,
        converge_threshold: float = 0.82,
    ) -> Optional[AnalysisResult]:
        """Run multi-agent analysis on extraction result.

        Args:
            extraction_result: Result from TrueUnifiedExtractor
            question: Analysis question to answer
            background: Optional background context
            max_rounds: Maximum analysis rounds
            converge_threshold: Convergence threshold

        Returns:
            Analysis result or None if disabled/failed
        """
        raise NotImplementedError

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if analysis is enabled."""
        raise NotImplementedError


__all__ = [
    "EvidencePlanePort",
    "HookResult",
    "HookRunnerPort",
    "IdentityRegistryPort",
    "AnalysisFinding",
    "AnalysisResult",
    "AnalysisResultPort",
]
