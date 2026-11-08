"""Analysis Models - 分析运行态领域模型

合并了以下小模型文件：
- analysis_status.py
- source_profile.py
- analysis_artifacts.py
- analysis_context.py
- analysis_result_metadata.py
- cleaned_document.py
- llm_usage.py
- pipeline_execution_result.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class AnalysisStatus(str, Enum):
    """分析状态枚举"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class SourceProfile:
    has_excel: bool = False
    has_doc: bool = False
    has_pdf: bool = False
    primary_type: str = "doc"

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any] | None) -> "SourceProfile":
        if not data:
            return cls()
        return cls(
            has_excel=data.get("has_excel", False),
            has_doc=data.get("has_doc", False),
            has_pdf=data.get("has_pdf", False),
            primary_type=data.get("primary_type", "doc"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_excel": self.has_excel,
            "has_doc": self.has_doc,
            "has_pdf": self.has_pdf,
            "primary_type": self.primary_type,
        }


@dataclass
class AnalysisArtifacts:
    manifest: str | None = None
    l1: str | None = None
    l2: str | None = None
    report: str | None = None
    core_results_md: str | None = None

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any] | None) -> "AnalysisArtifacts":
        if not data:
            return cls()
        return cls(
            manifest=data.get("manifest"),
            l1=data.get("l1"),
            l2=data.get("l2"),
            report=data.get("report"),
            core_results_md=data.get("core_results_md"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in {
                "manifest": self.manifest,
                "l1": self.l1,
                "l2": self.l2,
                "report": self.report,
                "core_results_md": self.core_results_md,
            }.items() if v is not None
        }


@dataclass
class AnalysisContext:
    run_id: str
    topic_id: str
    folder_path: str
    source_profile: SourceProfile
    fast_mode: bool
    smoke_mode: bool
    effective_max_files: int | None
    execution_warnings: list[str]
    discovered_documents: list[str]
    discovered_document_count: int
    runtime_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any] | None) -> "AnalysisContext":
        if not data:
            data = {}
        
        return cls(
            run_id=data.get("run_id", ""),
            topic_id=data.get("topic_id", ""),
            folder_path=data.get("folder_path", ""),
            source_profile=SourceProfile.from_legacy_dict(data.get("source_profile")),
            fast_mode=data.get("fast_mode", False),
            smoke_mode=data.get("smoke_mode", False),
            effective_max_files=data.get("effective_max_files"),
            execution_warnings=data.get("execution_warnings", []),
            discovered_documents=data.get("discovered_documents", []),
            discovered_document_count=data.get("discovered_document_count", 0),
            runtime_context=data.get("runtime_context") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "topic_id": self.topic_id,
            "folder_path": self.folder_path,
            "source_profile": self.source_profile.to_dict(),
            "fast_mode": self.fast_mode,
            "smoke_mode": self.smoke_mode,
            "effective_max_files": self.effective_max_files,
            "execution_warnings": self.execution_warnings,
            "discovered_documents": self.discovered_documents,
            "discovered_document_count": self.discovered_document_count,
            "runtime_context": self.runtime_context,
        }


@dataclass
class LLMUsage:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    cost: float = 0.0
    cost_estimate: float = 0.0

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any] | None) -> "LLMUsage":
        if not data:
            return cls()

        def _as_int(value: Any) -> int:
            try:
                return int(float(value))
            except Exception:
                return 0

        def _as_float(value: Any) -> float:
            try:
                return float(value)
            except Exception:
                return 0.0

        return cls(
            total_tokens=_as_int(data.get("total_tokens", data.get("tokens", 0))),
            prompt_tokens=_as_int(data.get("prompt_tokens", data.get("input_tokens", 0))),
            completion_tokens=_as_int(data.get("completion_tokens", data.get("output_tokens", 0))),
            llm_calls=_as_int(data.get("llm_calls", data.get("calls", 0))),
            cost=_as_float(data.get("cost", data.get("total_cost", data.get("cost_estimate", 0.0)))),
            cost_estimate=_as_float(data.get("cost_estimate", data.get("cost", data.get("total_cost", 0.0)))),
        )

    def to_dict(self, *, include_extended: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_estimate": self.cost_estimate,
        }

        if include_extended:
            if self.llm_calls:
                payload["llm_calls"] = self.llm_calls
            if self.cost:
                payload["cost"] = self.cost
        return payload


@dataclass
class AnalysisResultMetadata:
    status: AnalysisStatus
    run_id: str = ""
    manifest: str = ""
    run_dir: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    llm_usage: LLMUsage = field(default_factory=LLMUsage)
    quality: "QualityReport" = field(default_factory=lambda: None)
    artifacts: AnalysisArtifacts = field(default_factory=AnalysisArtifacts)

    @classmethod
    def from_legacy_dict(cls, data: dict[str, Any] | None) -> "AnalysisResultMetadata":
        if not data:
            return cls(status=AnalysisStatus.UNKNOWN)
        
        status_str = data.get("status", "UNKNOWN")
        try:
            status = AnalysisStatus(status_str)
        except ValueError:
            status = AnalysisStatus.UNKNOWN

        from .quality_report import QualityReport

        return cls(
            status=status,
            run_id=data.get("run_id", ""),
            manifest=data.get("manifest", ""),
            run_dir=data.get("run_dir", ""),
            summary=data.get("summary") or {},
            llm_usage=LLMUsage.from_legacy_dict(data.get("llm_usage")),
            quality=QualityReport.from_legacy_dict(data.get("quality")),
            artifacts=AnalysisArtifacts.from_legacy_dict(data.get("artifacts")),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status.value,
            "run_id": self.run_id,
            "manifest": self.manifest,
            "run_dir": self.run_dir,
            "summary": self.summary,
            "llm_usage": self.llm_usage.to_dict(),
            "artifacts": self.artifacts.to_dict(),
        }
        if self.quality:
            result["quality"] = self.quality.to_dict()
        return result


@dataclass(frozen=True)
class CleanedDocument:
    """Represents the result of extracting and readability-cleaning a document.

    Attributes:
        source_path: Input document path.
        source_type: File type (lowercase suffix), e.g. ".pdf".
        raw_text: Extracted raw text from the document.
        cleaned_text: LLM-refined text that improves readability while preserving details.
        warnings: Non-fatal warnings produced during processing.
        metadata: Non-sensitive metadata such as chunk counts and lengths.
        processing_timestamp: ISO-8601 timestamp when the document was cleaned (for auditability).
        cleaning_model_version: Identifier of the LLM/model version used for cleaning (for traceability).
    """

    source_path: str
    source_type: str
    raw_text: str
    cleaned_text: str
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    processing_timestamp: str | None = None
    cleaning_model_version: str | None = None


class DocumentProcessingError(RuntimeError):
    """Raised when a document cannot be processed into a cleaned output."""


class DocumentExtractionError(DocumentProcessingError):
    """Raised when raw text cannot be extracted from the document."""


class DocumentCleaningError(DocumentProcessingError):
    """Raised when the LLM cleaning step fails."""


@dataclass
class PipelineStageResult:
    """Typed result for a single pipeline stage execution."""

    stage: Optional[str] = None
    status: str = "SUCCESS"
    duration_ms: int = 0
    command: Optional[List[str]] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    action_hint: Optional[str] = None
    dispatch: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the stage result into a JSON-friendly dictionary."""
        return asdict(self)


@dataclass
class PipelineExecutionResult:
    """Typed result container for a multi-stage pipeline execution."""

    stages: List[PipelineStageResult] = field(default_factory=list)
    overall_status: str = "SUCCESS"

    def to_summary_payload(self) -> Dict[str, Any]:
        """Serialize the execution result into the orchestrator summary shape."""
        serialized_results: List[Dict[str, Any]] = []
        for stage in self.stages:
            if hasattr(stage, "to_dict"):
                serialized_results.append(stage.to_dict())
            elif is_dataclass(stage):
                serialized_results.append(asdict(stage))
            else:
                serialized_results.append(dict(stage))

        return {
            "overall_status": self.overall_status,
            "results": serialized_results,
        }
