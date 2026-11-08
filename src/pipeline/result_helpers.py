"""Result-payload helper functions extracted from MainPipeline.

Phase 1 of the main_pipeline LOC reduction (docs/CODE_CLEANUP_CANDIDATES.md
§3.1): pure helpers moved verbatim from MainPipeline methods. Behavior stays
locked by tests/pipeline/test_main_pipeline_characterization.py, which goes
through the delegating methods on MainPipeline. No orchestration state lives
here — every function takes explicit inputs.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from src.domain.models.core import ValidationCheck, ValidationReport

if TYPE_CHECKING:  # pragma: no cover
    from .main_pipeline import PipelineStatus as _PipelineStatus  # noqa: F401


def _is_pipeline_success(status: Any) -> bool:
    """Pipeline success check without importing PipelineStatus at runtime.

    Avoids circular ``main_pipeline -> result_helpers -> main_pipeline`` import.
    """
    return str(getattr(status, "value", status)) == "success"


def config_fingerprint(env: "os._Environ | dict") -> str:
    """Short deterministic fingerprint of the env knobs that affect pipeline
    behavior, for run replayability (A-3). Secrets are excluded by name.
    Changes to any included knob invalidate the fingerprint (visible in the
    manifest rather than silently baking into an old cache path)."""
    keys = [
        "LLM_PROVIDER",
        "LLM_OFFLINE_MODE",
        "LLM_MAX_INPUT_CHARS",
        "LLM_MAX_JSON_RETRIES",
        "LLM_RETRY_DELAY_SECONDS",
        "DEEPSEEK_TIMEOUT_SECONDS",
        "DIFY_TIMEOUT_SECONDS",
        "EXTRACT_FILE_TIMEOUT",
        "EXTRACT_CHUNK_MAX_RETRIES",
        "PROMPT_MAX_LENGTH",
        "PROMPT_VERSION",
        "SCHEMA_VERSION",
        "SKILL_VERSION",
        "LLM_CACHE_VERSION_BUNDLE",
    ]
    parts = []
    for key in sorted(keys):
        parts.append(f"{key}={env.get(key, '')}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:12]


# Historical name kept so existing callers/patches keep working.
_config_fingerprint = config_fingerprint


def build_resolved_config_snapshot(context) -> Dict[str, Any]:
    """Build a resolved-config snapshot for run replayability.

    2026-08-15 (A-3): every run must record the effective LLM / prompt /
    cache configuration so a later failure or benchmark can be replayed.
    Only env/config that affects *behavior* is included; secrets are
    excluded by name (never log API keys).
    """
    env = os.environ
    llm = {
        "provider": env.get("LLM_PROVIDER", "dify"),
        "fallback_providers": env.get("LLM_FALLBACK_PROVIDERS", ""),
        "offline_mode": env.get("LLM_OFFLINE_MODE", "0"),
        "model": env.get("DEEPSEEK_MODEL") or env.get("DIFY_MODEL", ""),
        "base_url": env.get("DEEPSEEK_BASE_URL", ""),
        "max_input_chars": env.get("LLM_MAX_INPUT_CHARS", "80000"),
        "max_json_retries": env.get("LLM_MAX_JSON_RETRIES", "2"),
        "retry_delay_seconds": env.get("LLM_RETRY_DELAY_SECONDS", "3"),
        "deepseek_timeout_seconds": env.get("DEEPSEEK_TIMEOUT_SECONDS", "180"),
        "dify_timeout_seconds": env.get("DIFY_TIMEOUT_SECONDS", "300"),
    }
    budget = {
        "extract_file_timeout": env.get("EXTRACT_FILE_TIMEOUT", "900"),
        "extract_chunk_max_retries": env.get("EXTRACT_CHUNK_MAX_RETRIES", "1"),
        "prompt_max_length": env.get("PROMPT_MAX_LENGTH", "12000"),
    }
    versions = {
        "schema_version": context.schema_version or env.get("SCHEMA_VERSION", ""),
        "skill_version": context.skill_version or env.get("SKILL_VERSION", ""),
        "prompt_version": context.prompt_version or env.get("PROMPT_VERSION", ""),
        "cache_version_bundle": env.get("LLM_CACHE_VERSION_BUNDLE", ""),
        "config_fingerprint": config_fingerprint(env),
    }
    return {"llm": llm, "budget": budget, "versions": versions}


def build_contract_artifacts(save_dir: str, validation_summary: Dict[str, Any]) -> Dict[str, str]:
    return {
        "pipeline_execution_result": str(Path(save_dir) / "pipeline_execution_result.contract.json"),
        "validation_report": str(Path(save_dir) / "validation_report.contract.json"),
        "artifact_envelope": str(Path(save_dir) / "artifact_envelope.contract.json"),
        "validation_state": validation_summary.get("validation_state", "pending"),
    }


def build_validation_report(pipeline, result) -> "ValidationReport":
    """Build the pipeline-level ValidationReport from a run result.

    P0-3 batch 3 (2026-09-09): moved verbatim from
    ``MainPipeline._build_validation_report`` using the established
    ``pipeline``-first-parameter pattern. Pure assembly: the four checks read
    only ``result``; the report identity/provenance come from ``pipeline``.
    """
    checks = [
        ValidationCheck(
            check_id="pipeline_publishable",
            name="Pipeline publishable state",
            status="passed" if result.is_publishable else "failed",
            message="Pipeline result is publishable" if result.is_publishable else "Pipeline result is not publishable",
            severity="info" if result.is_publishable else "error",
        ),
        ValidationCheck(
            check_id="pipeline_success",
            name="Pipeline success state",
            status="passed" if _is_pipeline_success(result.status) else "failed",
            message="Pipeline completed successfully" if _is_pipeline_success(result.status) else "Pipeline completed with issues",
            severity="info" if _is_pipeline_success(result.status) else "error",
        ),
        ValidationCheck(
            check_id="error_count",
            name="Pipeline error count",
            status="passed" if not result.errors else "failed",
            message=f"{len(result.errors)} errors detected",
            severity="error" if result.errors else "info",
        ),
        ValidationCheck(
            check_id="warning_count",
            name="Pipeline warning count",
            status="passed" if not result.warnings else "skipped",
            message=f"{len(result.warnings)} warnings detected",
            severity="warning" if result.warnings else "info",
        ),
    ]
    summary = {
        "passed": sum(1 for check in checks if check.status == "passed"),
        "failed": sum(1 for check in checks if check.status == "failed"),
        "skipped": sum(1 for check in checks if check.status == "skipped"),
    }
    return ValidationReport(
        report_id=f"validation-{pipeline._run_id}",
        target_ref=pipeline._run_id,
        target_type="pipeline_execution_result",
        status="passed" if result.is_publishable else "failed",
        validation_state="passed" if result.validation_passed else "failed",
        checks=checks,
        summary=summary,
        trace_context=pipeline._build_trace_context(artifact_id=f"validation-{pipeline._run_id}"),
        provenance=pipeline._build_provenance(),
        metadata=dict(pipeline.context.metadata),
    )


def deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去除重复实体，基于(layer, name, meta_class)进行语义去重

    修复L0自动补充节点跨文件重复问题：每个文件都会生成相同的L0基类，
    需要通过语义信息去重而不是仅依赖meta_id。
    """
    seen = {}
    unique = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        layer = entity.get("layer", "L1")
        name = entity.get("name", "").strip()
        meta_class = entity.get("meta_class", "").strip()

        # 对于L0和L1层，基于(layer, name, meta_class)去重
        # 对于其他层，增加source_file区分
        if layer in ("L0", "L1"):
            entity_key = (layer, name, meta_class)
        else:
            source_file = entity.get("source_file", "") or entity.get("evidence", {}).get("source_file", "")
            entity_key = (layer, name, meta_class, source_file) if source_file else (layer, name, meta_class)

        if entity_key not in seen:
            seen[entity_key] = entity
            unique.append(entity)

    if len(entities) != len(unique):
        print(f"[AssetPacket] Deduplicated entities: {len(entities)} -> {len(unique)}")
    return unique


def deduplicate_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去除重复关系，基于 source-target-type"""
    seen = set()
    unique = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        source = rel.get("source") or rel.get("source_id", "")
        target = rel.get("target") or rel.get("target_id", "")
        rel_type = rel.get("type") or rel.get("relation_type", "")
        key = f"{source}_{target}_{rel_type}"
        if key and key not in seen:
            seen.add(key)
            unique.append(rel)
    if len(relations) != len(unique):
        print(f"[AssetPacket] Deduplicated relations: {len(relations)} -> {len(unique)}")
    return unique


def extract_evidence_from_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从实体中提取证据数据"""
    evidence_list = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        ev = entity.get("evidence")
        if ev and isinstance(ev, dict):
            entity_id = entity.get("id") or entity.get("meta_id", "")
            if not entity_id:
                continue

            evidence_list.append({
                "id": ev.get("evidence_id", f"ev_{hash(str(ev)) % 1000000}"),
                "content": ev.get("content", ev.get("raw_text", "")),
                "source_file": ev.get("source_file", ""),
                "source_line": ev.get("line_range", [0])[0] if ev.get("line_range") else 0,
                "source_page": ev.get("page_range", [0])[0] if ev.get("page_range") else 0,
                "entity_refs": [entity_id],
                "relation_refs": [],
                "confidence": ev.get("confidence", 0.0),
                "chunk_id": "",
                "raw_text": ev.get("raw_text", ""),
                "context": ev.get("context", ""),
            })
    return evidence_list


def build_manifest_payload(
    pipeline,
    result: Any,
    result_dict: Dict[str, Any],
    contract_artifacts: Dict[str, str],
) -> Dict[str, Any]:
    """Assemble the run-manifest payload written next to the pipeline result.

    P0-3 batch 4 (2026-09-09): moved verbatim from
    ``MainPipeline._build_manifest_payload`` using the established
    ``pipeline``-first-parameter pattern. Every run manifest carries LLM
    observability stats (D-4) plus the resolved-config snapshot (A-3) so
    runs are replayable and comparable.
    """
    from src.application.use_cases.llm_observability_use_case import collect_llm_stats
    try:
        from src.application.services.llm_monitor import get_llm_monitor
        llm_stats = collect_llm_stats(get_llm_monitor())
    except Exception:
        llm_stats = {"total_calls": 0, "total_tokens": 0, "total_cost_estimate": 0.0, "stage_metrics": {}}
    return {
        "run_id": result.run_id,
        "source_path": pipeline.context.source_path,
        "domain": pipeline.context.domain,
        "status": result_dict.get("status"),
        "output_files": result_dict.get("output_files", []),
        "artifact_bundles": pipeline.context.metadata.get("artifact_bundles", []),
        "contract_artifacts": contract_artifacts,
        "retrieval_asset_packet": pipeline.context.metadata.get("retrieval_asset_packet_path", ""),
        "trace_id": pipeline._build_trace_context().trace_id,
        "resolved_config": pipeline._build_resolved_config_snapshot(),
        "stage_decision_trace": pipeline.context.metadata.get("stage_decision_trace", []),
        "llm_stats": llm_stats,
    }


def save_result_bundle(
    pipeline,
    result: Any,
    output_dir: "str | None" = None,
) -> str:
    """Persist the compatibility result plus standardized contract artifacts.

    P0-3 batch 4 (2026-09-09): moved verbatim from ``MainPipeline.save_result``
    using the established ``pipeline``-first-parameter pattern. Returns the
    result path. ``MainPipeline.save_result`` stays as a thin delegate (used
    by ``run_main_pipeline`` and any instance-level patches).
    """
    import logging

    logger = logging.getLogger(__name__)
    save_dir = output_dir or pipeline.context.output_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    result_dict = result.to_dict()
    core_result = pipeline._build_core_pipeline_result(result)
    validation_report = pipeline._build_validation_report(result)
    artifact_envelope = pipeline._build_artifact_envelope(result)
    contract_artifacts = pipeline._build_contract_artifacts(save_dir, validation_report)
    result_dict["contract_artifacts"] = contract_artifacts

    manifest_payload = build_manifest_payload(pipeline, result, result_dict, contract_artifacts)
    markdown_text = pipeline._pipeline_result_artifact_service.render_result_markdown(
        result_dict,
        ontology=pipeline.context.metadata.get("ontology_summary", {}) or {},
    )
    paths = pipeline._pipeline_result_artifact_service.persist_result_bundle(
        save_dir=save_dir,
        result=result,
        result_dict=result_dict,
        core_result=core_result,
        validation_report=validation_report,
        artifact_envelope=artifact_envelope,
        contract_artifacts=contract_artifacts,
        manifest_payload=manifest_payload,
        markdown_text=markdown_text,
    )

    logger.info(f"✅ Pipeline result saved to: {paths['result_path']}")
    logger.info(f"✅ Pipeline manifest saved to: {paths['manifest_path']}")
    logger.info(f"✅ Pipeline contract result saved to: {paths['core_result_path']}")
    logger.info(f"✅ Validation report saved to: {paths['validation_report_path']}")
    logger.info(f"✅ Artifact envelope saved to: {paths['artifact_envelope_path']}")
    logger.info(f"✅ Pipeline markdown saved to: {paths['markdown_path']}")
    return paths['result_path']


def build_core_pipeline_result(pipeline, result) -> "CorePipelineExecutionResult":
    """Build the canonical core pipeline execution result.

    P0-3 batch 10 (2026-09-09): moved verbatim from
    ``MainPipeline._build_core_pipeline_result`` using the established
    ``pipeline``-first-parameter pattern. Pure assembly over ``result`` plus
    pipeline trace/provenance/metadata.
    """
    from src.domain.models.core import (
        PipelineExecutionResult as CorePipelineExecutionResult,
        PipelineStageResult as CorePipelineStageResult,
    )

    stage_results = []
    for stage_name, stage_result in result.stage_results.items():
        stage_results.append(
            CorePipelineStageResult(
                stage=stage_name,
                status="success" if stage_result.success else "failed",
                duration_ms=int(stage_result.elapsed_seconds * 1000),
                warnings=stage_result.data.get("warnings", []) if stage_result.data else [],
                metrics=stage_result.data.get("metrics", {}) if stage_result.data else {},
                error_type=stage_result.data.get("error_type") if stage_result.data else None,
                action_hint=stage_result.data.get("action_hint") if stage_result.data else None,
                dispatch=stage_result.data.get("dispatch", {}) if stage_result.data else {},
                message=stage_result.message,
            )
        )

    trace_context = pipeline._build_trace_context(artifact_id=f"pipeline-result-{pipeline._run_id}")
    return CorePipelineExecutionResult(
        run_id=result.run_id,
        pipeline_name="main_pipeline",
        pipeline_version=pipeline.context.metadata.get("pipeline_version", "1.0.0"),
        overall_status=result.status.value.upper(),
        stages=stage_results,
        source_path=pipeline.context.source_path,
        domain=pipeline.context.domain,
        entity_count=result.stats.entity_count,
        relation_count=result.stats.relation_count,
        elapsed_seconds=result.stats.total_latency_ms / 1000.0,
        errors=result.errors,
        warnings=result.warnings,
        trace_context=trace_context,
        artifact_refs=list(result.output_files),
        validation={
            "status": "passed" if result.validation_passed else "failed",
            "warnings": len(result.warnings),
            "publishable": result.is_publishable,
            "quality_status": result.quality_status.value,
            "quality_reason": result.quality_reason,
            "quality_score": result.quality_score,
        },
        cache_stats={"llm_calls": result.stats.llm_calls},
        provenance=pipeline._build_provenance(),
        metadata=dict(pipeline.context.metadata),
    )


def build_artifact_envelope(pipeline, result, core_result) -> "CoreArtifactEnvelope":
    """Build the core artifact envelope for a pipeline result.

    P0-3 batch 10 (2026-09-09): moved verbatim from
    ``MainPipeline._build_artifact_envelope`` using the established
    ``pipeline``-first-parameter pattern. ``core_result`` is passed in (the
    MainPipeline delegate chains through its own seam) so direct-instance
    callers keep working.
    """
    from src.domain.models.core import ArtifactEnvelope as CoreArtifactEnvelope

    return CoreArtifactEnvelope(
        artifact_id=f"pipeline-result-{pipeline._run_id}",
        artifact_type="pipeline_execution_result",
        artifact_version=core_result.contract_version,
        status="stable" if core_result.overall_status == "SUCCESS" and result.is_publishable else "draft",
        trace_context=pipeline._build_trace_context(artifact_id=f"pipeline-result-{pipeline._run_id}"),
        provenance=core_result.provenance,
        quality={
            "overall_status": core_result.overall_status,
            "publishable": result.is_publishable,
            "quality_status": result.quality_status.value,
            "quality_reason": result.quality_reason,
            "quality_score": result.quality_score,
        },
        payload=core_result.model_dump(exclude_none=True),
        relationships=[],
        metadata={"run_id": pipeline._run_id, **pipeline._build_run_metadata()},
    )
