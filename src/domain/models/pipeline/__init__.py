"""Pipeline domain models.

Stable domain grouping for pipeline execution result types.

Canonical owner: `src.domain.models.core` (ContractModel). This subpackage is a
stable re-export of the canonical contracts. The legacy dataclasses in
`src.domain.models.analysis_models` remain importable directly for the stage
execution use cases that still rely on them (Phase 4 migration in progress).
"""
from ..core import PipelineExecutionResult, PipelineStageResult

__all__ = [
    "PipelineStageResult",
    "PipelineExecutionResult",
]
