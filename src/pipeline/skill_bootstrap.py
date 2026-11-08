"""Skill bootstrap + post-execution validation extracted from MainPipeline.

P0-3 batch 16: skill-integration bootstrap and the post-execution validation
cluster, moved verbatim from MainPipeline methods using the established
``pipeline``-first-parameter pattern (same as stage_skills, runflow,
status_quality, stage_plumbing, trace_provenance, regression_metadata).

Contents:
- :func:`initialize_skill_integration` — from
  ``_initialize_skill_integration``
- :func:`run_post_execution_validation` — from
  ``_run_post_execution_validation``
- :func:`validate_pipeline_output` — B3 use-case wiring, from
  ``_validate_pipeline_output``
- :func:`run_quality_gate` — B1 use-case wiring, from ``_run_quality_gate``

Behavior stays locked by tests/pipeline/test_main_pipeline_characterization.py
(``_run_post_execution_validation`` is patched at instance level in the
run-flow tests; the MainPipeline delegate preserves that seam).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline

logger = logging.getLogger(__name__)


def initialize_skill_integration(pipeline: "MainPipeline") -> None:
    """Bootstrap the skill bridge, router and contract validator.

    1. Load the skill bridge
    2. Route applicable skills (config/pipeline_skill_routing.json)
    3. Initialize the skill contract validator (UnifiedSkillContract)
    """
    try:
        from src.pipeline.pipeline_skill_router import get_pipeline_skill_router
        from skills.skill_contract import UnifiedSkillContractValidator

        bridge_stats = pipeline._skill_pipeline_bridge.get_stats()
        logger.info(f"🔧 Skill Pipeline Bridge initialized: {bridge_stats['skills_count']} skills loaded")

        pipeline._pipeline_skill_router = get_pipeline_skill_router()
        extract_skills = pipeline._pipeline_skill_router.get_skills_for_stage("extract", pipeline.context.domain)
        logger.info(f"🎯 Active skills for domain {pipeline.context.domain}: {extract_skills}")

        default_contract = pipeline._build_default_skill_contract()
        pipeline._skill_contract_validator = UnifiedSkillContractValidator(default_contract)
        logger.info("✅ Unified Skill contract validator initialized")
    except Exception as e:
        logger.warning(f"Skill integration initialization skipped: {e}")


def run_post_execution_validation(pipeline: "MainPipeline") -> None:
    """Run post-execution validation (output validation + quality gate)."""
    try:
        output_data = pipeline._collect_pipeline_output()
        if output_data:
            validate_pipeline_output(pipeline, output_data)
            run_quality_gate(pipeline, output_data)
            if isinstance(pipeline.context.metadata, dict):
                pipeline.context.metadata.setdefault("quality_gate", {})
                pipeline.context.metadata["quality_gate"].update({
                    "output_validation": "pending",
                    "source": "run_post_execution_validation",
                })
    except Exception as e:
        logger.warning(f"Post-execution validation skipped due to error: {e}")


def validate_pipeline_output(pipeline: "MainPipeline", output_data: Dict[str, Any]) -> None:
    """Validate assembled output; wire error/warning chips into state."""
    # B3 (2026-08-15): output validation migrated to the pipeline_validation
    # use case; this function wires error/warning chips into state.
    from src.application.use_cases.pipeline_validation_use_case import (
        validate_pipeline_output as _validate,
    )
    from src.infrastructure.config.contract_validation import get_validation_chain

    outcome = _validate(get_validation_chain(), output_data)
    pipeline._errors.extend(outcome.error_chips)
    pipeline._warnings.extend(outcome.warning_chips)


def run_quality_gate(pipeline: "MainPipeline", output_data: Dict[str, Any]) -> None:
    """Run the governance audit; wire results into errors/warnings/metadata."""
    # B1 (2026-08-15): governance audit + metadata assembly migrated to the
    # quality_gate use case; this function only wires results into state.
    from src.application.use_cases.quality_gate_use_case import (
        run_quality_gate as _run_gate,
    )

    outcome = _run_gate(output_data)
    pipeline._errors.extend(outcome.fatal_errors)
    pipeline._warnings.extend(outcome.warning_messages)

    pipeline._stats.metadata = outcome.metadata
    if isinstance(pipeline.context.metadata, dict):
        pipeline.context.metadata.setdefault("quality_gate", {})
        pipeline.context.metadata["quality_gate"].update(outcome.metadata)
