"""Stage-result builders (P1-3, 2026-10-16).

Pure constructors extracted verbatim from ``MainPipeline``:

- :func:`make_skipped_stage_result` — ``StageDecision`` SKIP result
- :func:`make_degraded_stage_result` — ``StageDecision`` DEGRADE result
- :func:`skip_reason_for_mode` — non-production skip predicate

``MainPipeline`` keeps thin delegates (``_make_skipped_stage_result`` /
``_make_degraded_stage_result`` / ``_should_skip_stage``) because
``stage_execution_core`` calls them as instance methods (``pipeline._make_*``).
Side effects (logging + a warnings sink) are passed in explicitly so the
functions stay free of pipeline state.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .stage_decision import StageDecision
from .stages.common import StageResult
from .pipeline_models import PipelineExecutionMode, PipelineStage

logger = logging.getLogger(__name__)


def make_skipped_stage_result(
    stage: PipelineStage,
    *,
    reason: str,
    decision: "StageDecision",
    run_id: str,
    warnings: List[str],
) -> StageResult:
    """Build the ``StageResult`` for a ``StageDecision``-driven skip."""
    decision_value = decision.value if isinstance(decision, StageDecision) else str(decision)
    logger.info(f"⏭️  StageDecision.{decision_value}: {stage.value} — {reason}")
    warnings.append(f"Skipped stage {stage.value} ({decision_value}): {reason}")
    return StageResult(
        stage=stage.value,
        success=True,
        message=f"[{decision_value}] Skipped. Reason: {reason}",
        data={
            "skipped": True,
            "stage_decision": decision_value,
            "reason": reason,
            "source_hash": run_id,
        },
        elapsed_seconds=0.0,
        run_dir="",
        fragment_path="",
    )


def make_degraded_stage_result(
    stage: PipelineStage,
    *,
    reason: str,
    run_id: str,
    warnings: List[str],
) -> StageResult:
    """Build the ``StageResult`` for a DEGRADE decision (LLM path skipped)."""
    logger.warning(f"🟡 StageDecision.DEGRADE: {stage.value} — {reason}")
    warnings.append(f"Degraded stage {stage.value}: {reason}")
    return StageResult(
        stage=stage.value,
        success=True,
        message=f"[DEGRADE] Skipped LLM-dependent path. Reason: {reason}",
        data={
            "skipped": True,
            "stage_decision": "DEGRADE",
            "reason": reason,
            "quality_level": "DEGRADED",
            "source_hash": run_id,
        },
        elapsed_seconds=0.0,
        run_dir="",
        fragment_path="",
    )


def skip_reason_for_mode(
    execution_mode: PipelineExecutionMode,
    stage: PipelineStage,
) -> Optional[str]:
    """Return a skip reason when *stage* is disabled in *execution_mode*.

    Only non-PRODUCTION modes skip anything today (emergent reasoning / guard),
    matching the historical ``_should_skip_stage`` behavior.
    """
    if execution_mode != PipelineExecutionMode.PRODUCTION and stage in {
        PipelineStage.EMERGENT_REASONING,
        PipelineStage.GUARD,
    }:
        return f"Skipped stage {stage.value} in {execution_mode.value} mode"
    return None
