"""Stage-skill hooks extracted from MainPipeline (P0-3 batch 9).

Per-stage skill routing, execution, contract validation and the default skill
contract, moved verbatim from MainPipeline methods using the established
``pipeline``-first-parameter pattern (same as stage_plumbing,
trace_provenance, regression_metadata, result_helpers).

Contents:
- :func:`build_default_skill_contract` — from
  ``_build_default_skill_contract``
- :func:`apply_stage_skills` — from ``_apply_stage_skills`` (async)
- :func:`get_skills_for_stage` — from ``_get_skills_for_stage``
- :func:`merge_skill_result` — from ``_merge_skill_result``
- :func:`validate_stage_output` — from ``_validate_stage_output``
- :func:`collect_validation_entities` — from
  ``_collect_validation_entities``
- :func:`collect_validation_relations` — from
  ``_collect_validation_relations``
- :func:`record_stage_validation_violations` — from
  ``_record_stage_validation_violations``

Behavior stays locked by tests/pipeline/test_main_pipeline_characterization.py
(the run-flow tests patch ``_apply_stage_skills`` / ``_validate_stage_output``
at instance level as seams; the delegates on MainPipeline preserve them).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from skills.skill_contract import (
    UnifiedSkillContract,
    ClassificationContract,
    EntityContract,
    EntityFieldRequirement,
    ContractSeverity,
    RelationContract,
)

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline
    from .stages import StageResult

logger = logging.getLogger(__name__)


def build_default_skill_contract(pipeline: "MainPipeline") -> UnifiedSkillContract:
    """Build the default pipeline skill contract for the run domain."""
    return UnifiedSkillContract(
        skill_id="default_pipeline_contract",
        name="Default Pipeline Contract",
        description="Default contract for pipeline execution",
        version="1.0.0",
        category="pipeline",
        business_domain=[pipeline.context.domain],
        supported_layers=["L0", "L1", "L2", "L2.5"],
        classification_contract=ClassificationContract(
            name="Default Pipeline Classification",
            description="Default classification rules",
            applicable_domains=[pipeline.context.domain],
            applicable_document_types=["all"],
        ),
        entity_contracts=[
            EntityContract(
                entity_type="*",
                description="Generic entity validation rules",
                required_fields=[
                    EntityFieldRequirement(field_name="id", severity=ContractSeverity.REQUIRED),
                    EntityFieldRequirement(field_name="name", severity=ContractSeverity.REQUIRED),
                    EntityFieldRequirement(field_name="evidence", severity=ContractSeverity.REQUIRED),
                ],
                priority=1,
            ),
        ],
        relation_contracts=[
            RelationContract(
                relation_type="*",
                allowed=True,
                description="Default relation type",
                evidence_required=True,
                confidence_threshold=0.7,
            ),
        ],
    )


async def apply_stage_skills(pipeline: "MainPipeline", stage, stage_result: "StageResult") -> None:
    """Route and execute applicable skills against a stage result."""
    try:
        stage_skills = get_skills_for_stage(pipeline, stage)
        if not stage_skills:
            return

        logger.info(f"⚡ Applying skills to stage {stage.value}: {stage_skills}")

        stage_data = stage_result.data or {}
        for skill_id in stage_skills:
            try:
                skill_context = {
                    "stage": stage.value,
                    "run_id": pipeline._run_id,
                    "domain": pipeline.context.domain,
                    "stage_data": stage_data,
                }
                skill_result = pipeline._skill_pipeline_bridge.execute_skill(skill_id, skill_context)

                if skill_result.success:
                    logger.debug(f"Skill {skill_id} executed successfully")
                    merge_skill_result(pipeline, stage_result, skill_result)
                else:
                    logger.warning(f"Skill {skill_id} execution failed: {skill_result.errors}")
            except Exception as e:
                logger.error(f"Error applying skill {skill_id}: {e}")
    except Exception as e:
        logger.debug(f"Skill application skipped: {e}")


def get_skills_for_stage(pipeline: "MainPipeline", stage) -> List[str]:
    """Return the skill list for a stage (config-driven routing)."""
    if hasattr(pipeline, '_pipeline_skill_router') and pipeline._pipeline_skill_router:
        return pipeline._pipeline_skill_router.get_skills_for_stage(stage.value, pipeline.context.domain)

    return []


def merge_skill_result(pipeline: "MainPipeline", stage_result: "StageResult", skill_result: Any) -> None:
    """Merge a skill execution result into the stage result data."""
    if not stage_result.data:
        stage_result.data = {}

    skill_output = skill_result.output.get("output", skill_result.output)
    if isinstance(skill_output, dict):
        stage_result.data.update(skill_output)


def validate_stage_output(pipeline: "MainPipeline", stage, stage_result: "StageResult") -> None:
    """Validate stage output against the skill contract."""
    if not pipeline._skill_contract_validator:
        return

    try:
        stage_data = stage_result.data or {}
        entities = collect_validation_entities(pipeline, stage_data)
        relations = collect_validation_relations(pipeline, stage_data)

        if entities or relations:
            violations = pipeline._skill_contract_validator.validate(entities, relations)
            record_stage_validation_violations(pipeline, stage, violations)
    except Exception as e:
        logger.debug(f"Stage validation skipped: {e}")


def collect_validation_entities(pipeline: "MainPipeline", stage_data: Dict[str, Any]) -> Dict[str, Any]:
    """Collect ``*_entities`` buckets from stage data for contract validation."""
    entities = {}
    for key, value in stage_data.items():
        if isinstance(value, list) and key.endswith("_entities"):
            entities[key] = value
    return entities


def collect_validation_relations(pipeline: "MainPipeline", stage_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Collect the relation list from stage data for contract validation."""
    relations = stage_data.get("relationships", []) or stage_data.get("relations", [])
    return relations if isinstance(relations, list) else []


def record_stage_validation_violations(pipeline: "MainPipeline", stage, violations: list[Dict[str, Any]]) -> None:
    """Record contract violations as pipeline warnings (first 3 surfaced)."""
    if not violations:
        return
    summary = pipeline._skill_contract_validator.get_violation_summary(violations)
    logger.warning(f"⚠️ Contract violations detected in {stage.value}: {summary}")
    for violation in violations[:3]:
        pipeline._warnings.append(
            f"Contract violation [{stage.value}]: {violation.get('field', '')} - {violation.get('message', '')}"
        )
