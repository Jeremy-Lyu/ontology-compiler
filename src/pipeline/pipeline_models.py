"""Pipeline context and result models (P0-3 batch 14).

Enums + dataclasses moved verbatim from main_pipeline so orchestration,
status/quality, runflow and the characterization tests share one canonical
home. main_pipeline keeps re-export aliases (router, scripts, tests and
``src.pipeline.__init__`` import these names from main_pipeline today).

Contents: PipelineStage / PipelineStatus / PipelineQualityStatus /
PipelineExecutionMode / PipelineContext / PipelineStats /
PipelineExecutionResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from .stages import StageResult


class PipelineStage(Enum):
    PREFLIGHT = "preflight"
    EXTRACT = "extract"
    MERGE = "merge"
    DEDUP = "dedup"
    GRAPH_FUSION = "graph_fusion"
    RELATIONSHIP = "relationship"
    AUDIT = "audit"
    DCS_AUDIT = "dcs_audit"
    GUARD = "guard"
    EMERGENT_REASONING = "emergent_reasoning"
    EXPORT = "export"
    DOCUMENT_REPORT = "document_report"
    SUMMARIZE = "summarize"


class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class PipelineQualityStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    DEGRADED_SUCCESS = "degraded_success"
    FAILURE = "failure"


class PipelineExecutionMode(Enum):
    PRODUCTION = "production"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class PipelineContext:
    source_path: str
    domain: str = "SUPPLY_CHAIN"
    run_id: str = ""
    output_dir: str = ""
    fast_mode: bool = False
    enable_rigor_extract: bool = False
    enable_llm_conflict_resolver: bool = True
    max_retries: int = 3
    regression_mode: bool = False
    execution_mode: str = PipelineExecutionMode.PRODUCTION.value
    execution_reason: str = ""
    source_hash: str = ""
    schema_version: str = ""
    skill_version: str = ""
    prompt_version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStats:
    total_stages: int = 0
    completed_stages: int = 0
    failed_stages: int = 0
    total_latency_ms: float = 0.0
    llm_calls: int = 0
    entity_count: int = 0
    relation_count: int = 0
    evidence_count: int = 0


@dataclass
class PipelineExecutionResult:
    run_id: str
    status: PipelineStatus
    context: PipelineContext
    stats: PipelineStats
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    publishable: bool = True
    execution_mode: str = PipelineExecutionMode.PRODUCTION.value
    execution_reason: str = ""
    quality_status: PipelineQualityStatus = PipelineQualityStatus.SUCCESS
    quality_reason: str = ""
    validation_passed: bool = False
    quality_score: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == PipelineStatus.SUCCESS

    @property
    def is_publishable(self) -> bool:
        return self.publishable and self.status == PipelineStatus.SUCCESS and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "quality_status": self.quality_status.value,
            "quality_reason": self.quality_reason,
            "validation_passed": self.validation_passed,
            "quality_score": self.quality_score,
            "context": {
                "source_path": self.context.source_path,
                "source_hash": self.context.source_hash,
                "schema_version": self.context.schema_version,
                "skill_version": self.context.skill_version,
                "prompt_version": self.context.prompt_version,
                "domain": self.context.domain,
                "output_dir": self.context.output_dir,
                "fast_mode": self.context.fast_mode,
                "enable_rigor_extract": self.context.enable_rigor_extract,
                "enable_llm_conflict_resolver": self.context.enable_llm_conflict_resolver,
                "regression_mode": self.context.regression_mode,
                "execution_mode": self.context.execution_mode,
            },
            "stats": {
                "total_stages": self.stats.total_stages,
                "completed_stages": self.stats.completed_stages,
                "failed_stages": self.stats.failed_stages,
                "total_latency_ms": self.stats.total_latency_ms,
                "llm_calls": self.stats.llm_calls,
                "entity_count": self.stats.entity_count,
                "relation_count": self.stats.relation_count,
                "evidence_count": self.stats.evidence_count,
            },
            "stage_results": {
                stage: result.to_dict() for stage, result in self.stage_results.items()
            },
            "output_files": self.output_files,
            "errors": self.errors,
            "warnings": self.warnings,
            "publishable": self.publishable,
            "execution_mode": self.execution_mode,
            "execution_reason": self.execution_reason,
            "metadata": self.context.metadata,
        }


