"""Application ports for dependency inversion.

This package defines all port interfaces used for application layer
dependency inversion. Ports are abstract interfaces that define
what the application needs, not how it's implemented.

Usage:
    from src.application.ports import OntologyExtractorPort, DeduplicationPort

    class MyExtractor(OntologyExtractorPort):
        async def extract(self, file_path: str, domain: str | None = None) -> ExtractorResult:
            ...

P2 port classification (§9.8): the ports are now split by bounded context
into the classified modules in this package (document_ports, extraction_ports,
artifact_ports, quality_ports, skill_ports, observability_ports, rag_ports,
llm_ports, folder_analysis_ports). The former single-file facade
`all_ports.py` has been deleted; this package re-exports the full facade
surface directly from those classified modules so `from src.application.ports
import X` keeps working identically. New code should import from the
classified module matching its bounded context instead.
"""
from __future__ import annotations

# Neutral compiler-facing contracts stay canonical in `core.contracts.ports`.
from core.contracts.ports import (
    ArtifactStorePort,
    CapsuleQueryPort,
    EvidencePlanePort,
    HookResult,
    HookRunnerPort,
    IdentityRegistryPort,
    PipelineSummaryPaths,
    StoredArtifact,
    StoredArtifactSet,
)

from .artifact_ports import (
    StoragePort,
)
from .document_ports import (
    DocumentContent,
    DocumentReaderPort,
    TextFragment,
)
from .extraction_ports import (
    ConflictDetectionPort,
    DeduplicationPort,
    ExtractorResult,
    OntologyExtractorPort,
)
from .folder_analysis_ports import (
    FolderAnalysisRequest,
    FolderAnalysisResult,
    FolderAnalysisServicePort,
    SessionReplayStorePort,
)
from .llm_ports import (
    LLMPort,
)
from .observability_ports import (
    AnalysisFinding,
    AnalysisResult,
    AnalysisResultPort,
)
from .quality_ports import (
    GovernanceValidatorPort,
    MergeDecision,
    MergeDecisionPort,
    PatchApplicationPort,
    PatchOperation,
    QualityAssessmentPort,
    QualityDimensions,
    ReportGenerationPort,
)
from .rag_ports import (
    RAGBackendUnavailableError,
    RAGFeedbackPort,
    RAGSearchError,
    RAGSearchPort,
    RAGSearchResult,
    RAGSearchTimeoutError,
)
from .skill_ports import (
    SkillExecutionResult,
    SkillManifest,
    SkillRegistryPort,
)

__all__ = [
    # Data classes
    "StoredArtifact",
    "StoredArtifactSet",
    "PipelineSummaryPaths",
    "TextFragment",
    "FolderAnalysisRequest",
    "FolderAnalysisResult",
    "HookResult",
    "ExtractorResult",
    "QualityDimensions",
    "MergeDecision",
    "PatchOperation",
    "RAGSearchResult",
    "AnalysisFinding",
    "AnalysisResult",
    "SkillManifest",
    "SkillExecutionResult",
    "DocumentContent",
    # Ports
    "ArtifactStorePort",
    "CapsuleQueryPort",
    "ConflictDetectionPort",
    "DeduplicationPort",
    "DocumentReaderPort",
    "EvidencePlanePort",
    "FolderAnalysisServicePort",
    "GovernanceValidatorPort",
    "HookRunnerPort",
    "IdentityRegistryPort",
    "LLMPort",
    "MergeDecisionPort",
    "OntologyExtractorPort",
    "PatchApplicationPort",
    "QualityAssessmentPort",
    "RAGFeedbackPort",
    "RAGSearchPort",
    "RAGSearchError",
    "RAGBackendUnavailableError",
    "RAGSearchTimeoutError",
    "ReportGenerationPort",
    "SessionReplayStorePort",
    "StoragePort",
    "AnalysisResultPort",
    "SkillRegistryPort",
]
