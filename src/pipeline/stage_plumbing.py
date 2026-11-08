"""Stage-plumbing helpers extracted from MainPipeline (P0-3 batch 7).

Thin wrappers around the extracted stage-* policy modules, moved verbatim from
MainPipeline methods using the established ``pipeline``-first-parameter pattern
(same as trace_provenance, result_helpers, regression_metadata).

Contents:
- :func:`normalize_artifact_path` — real-file-preserving path normalization
- :func:`sanitize_stage_result` — paths + relationships + entities + metadata
- :func:`normalize_stage_result_paths` — path-field normalization
- :func:`build_entity_layer_index` — entity_layer_policy delegation
- :func:`rewrite_stage_relationships` — relationship rewrite + layer-pair gate
- :func:`normalize_stage_entities` — entity normalization + profile summary
- :func:`normalize_entity_record` — entity_layer_policy delegation
- :func:`infer_entity_layer` — entity_layer_policy delegation
- :func:`build_entity_profile_summary` — entity_layer_policy delegation
- :func:`project_structured_metadata` — entity_layer_policy delegation
- :func:`is_layer_pair_blocked` — entity_layer_policy gate (state injected)
- :func:`lookup_entity_layer` — stage-results layer lookup
- :func:`rewrite_related_to` — relation_rewrite_policy delegation
- :func:`build_stage_args` — stage_args delegation
- :func:`resolve_stage_fragment_arg` — stages.common delegation
- :func:`collect_stats` / :func:`collect_output_files` — accumulator wiring
- :func:`log_summary` — run-summary logging

Behavior stays locked by tests/pipeline/test_main_pipeline_characterization.py
(via ``_normalize_artifact_path`` direct call, sanitizer side effects on stage
flows, and the end-to-end run/quality paths).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from ontology_compiler.services.relation_governance import get_relation_governance_engine

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import MainPipeline
    from .stages import StageResult

logger = logging.getLogger(__name__)


def normalize_artifact_path(pipeline: "MainPipeline", artifact_path: str) -> str:
    """Normalize a stage artifact path for downstream consumption.

    Keeps the original path when it actually exists. Only falls back to the
    flattened legacy layout (``.../fragment_X.json``) when the original does
    not exist, so extract's real fragment under ``.../files/0000/`` is not
    lost.
    """
    path = Path(artifact_path)
    if path.is_file():
        return str(path)
    parts = list(path.parts)
    if "files" in parts:
        first_files_idx = parts.index("files")
        flattened = Path(*parts[:first_files_idx]) / path.name
        if flattened.is_file():
            return str(flattened)
    return str(path)


def sanitize_stage_result(pipeline: "MainPipeline", stage_result: "StageResult") -> "StageResult":
    """Normalize paths, relationships, entities and project structured metadata."""
    if not stage_result.data:
        return stage_result
    normalize_stage_result_paths(pipeline, stage_result)
    rewrite_stage_relationships(pipeline, stage_result)
    normalize_stage_entities(pipeline, stage_result)
    if isinstance(stage_result.data.get("metadata"), dict):
        stage_result.data["structured_metadata"] = project_structured_metadata(
            pipeline, stage_result.data["metadata"]
        )
    return stage_result


def normalize_stage_result_paths(pipeline: "MainPipeline", stage_result: "StageResult") -> None:
    """Normalize path fields of a stage result in place."""
    for key in ("governed_fragment_path", "run_dir", "output_dir"):
        value = stage_result.data.get(key)
        if isinstance(value, str) and value:
            stage_result.data[key] = normalize_artifact_path(pipeline, value)
    fragment_value = stage_result.data.get("fragment_path")
    if isinstance(fragment_value, str) and fragment_value:
        stage_result.fragment_path = normalize_artifact_path(pipeline, fragment_value)
    artifact_paths = stage_result.data.get("artifact_paths")
    if isinstance(artifact_paths, list):
        stage_result.artifact_paths = [normalize_artifact_path(pipeline, p) for p in artifact_paths if isinstance(p, str) and p]
    if pipeline.context.regression_mode:
        stage_result.data["regression_mode"] = True


def build_entity_layer_index(pipeline: "MainPipeline", stage_result: "StageResult") -> dict[str, str]:
    """Build the entity-id → layer index for relationship rewriting."""
    from .entity_layer_policy import build_entity_layer_index as _build

    return _build(stage_result.data)


def rewrite_stage_relationships(pipeline: "MainPipeline", stage_result: "StageResult") -> None:
    """Rewrite RELATED_TO + annotate/drop layer-blocked relationships in place."""
    relationships = stage_result.data.get("relationships")
    if not isinstance(relationships, list):
        return
    id_to_layer = build_entity_layer_index(pipeline, stage_result)
    rewritten = []
    dropped_layer_pair = 0
    for rel in relationships:
        if not isinstance(rel, dict):
            rewritten.append(rel)
            continue
        rel_type = str(rel.get("type") or rel.get("relation_type") or "").upper().strip()
        if rel_type == "RELATED_TO":
            rel = rewrite_related_to(pipeline, rel)
        source_id = str(rel.get("source") or rel.get("from") or rel.get("source_id") or rel.get("from_id") or "")
        target_id = str(rel.get("target") or rel.get("to") or rel.get("target_id") or rel.get("to_id") or "")
        source_layer = str(rel.get("source_layer") or rel.get("from_layer") or id_to_layer.get(source_id, "") or "")
        target_layer = str(rel.get("target_layer") or rel.get("to_layer") or id_to_layer.get(target_id, "") or "")
        if source_layer and target_layer:
            rel["source_layer"] = source_layer
            rel["target_layer"] = target_layer
            if is_layer_pair_blocked(pipeline, rel):
                dropped_layer_pair += 1
                continue
        rewritten.append(rel)
    stage_result.data["relationships"] = rewritten
    stage_result.data["relationship_count"] = len(rewritten)
    if dropped_layer_pair:
        logger.info(f"🔧 Dropped {dropped_layer_pair} relationships with blocked layer pairs")


def normalize_stage_entities(pipeline: "MainPipeline", stage_result: "StageResult") -> None:
    """Normalize entity records and rebuild profile summaries in place."""
    if isinstance(stage_result.data.get("entities"), list):
        stage_result.data["entities"] = [normalize_entity_record(pipeline, item) for item in stage_result.data["entities"]]
        stage_result.data["entity_profile_summary"] = build_entity_profile_summary(pipeline, stage_result.data["entities"])
    if isinstance(stage_result.data.get("nodes"), list):
        stage_result.data["nodes"] = [normalize_entity_record(pipeline, item) for item in stage_result.data["nodes"]]
        stage_result.data.setdefault("entity_profile_summary", build_entity_profile_summary(pipeline, stage_result.data["nodes"]))


def normalize_entity_record(pipeline: "MainPipeline", entity: Any) -> Any:
    """Normalize a single entity record via the entity_layer_policy module."""
    from .entity_layer_policy import normalize_entity_record as _normalize
    from .stages.common import normalize_entity

    return _normalize(entity, normalize_fn=normalize_entity)


def infer_entity_layer(pipeline: "MainPipeline", entity: Dict[str, Any]) -> Dict[str, Any]:
    """Infer the layer of a single entity via the entity_layer_policy module."""
    from .entity_layer_policy import infer_entity_layer as _infer

    return _infer(entity)


def build_entity_profile_summary(pipeline: "MainPipeline", entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an entity profile summary via the entity_layer_policy module."""
    from .entity_layer_policy import build_entity_profile_summary as _build

    return _build(entities)


def project_structured_metadata(pipeline: "MainPipeline", metadata: dict[str, Any]) -> dict[str, Any]:
    """Project structured metadata via the entity_layer_policy module."""
    from .entity_layer_policy import project_structured_metadata as _project

    return _project(metadata)


def is_layer_pair_blocked(pipeline: "MainPipeline", rel: Dict[str, Any]) -> bool:
    """Check the layer-pair block gate (pipeline stage-results injected)."""
    from .entity_layer_policy import is_layer_pair_blocked as _blocked

    return _blocked(
        rel,
        lookup_layer=lambda entity_id: lookup_entity_layer(pipeline, entity_id),
        governance_engine=get_relation_governance_engine(),
    )


def lookup_entity_layer(pipeline: "MainPipeline", entity_id: str) -> str:
    """Look up entity layer from stage results by entity ID."""
    if not entity_id:
        return ""
    for stage_result in pipeline._stage_results.values():
        if not stage_result or not stage_result.data:
            continue
        for bucket_key in ("nodes", "entities", "instances"):
            bucket = stage_result.data.get(bucket_key)
            if not isinstance(bucket, list):
                continue
            for ent in bucket:
                if not isinstance(ent, dict):
                    continue
                eid = str(ent.get("meta_id") or ent.get("entity_id") or ent.get("id") or "")
                if eid == entity_id:
                    layer = str(ent.get("layer") or "")
                    if layer:
                        return layer
    return ""


def rewrite_related_to(pipeline: "MainPipeline", rel: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite a RELATED_TO relation via the relation_rewrite_policy module."""
    from .relation_rewrite_policy import rewrite_related_to as _rewrite

    return _rewrite(rel)


def build_stage_args(pipeline: "MainPipeline") -> argparse.Namespace:
    """Build the stage-execution Namespace via the stage_args module."""
    from .stage_args import build_stage_args as _build
    from .stages import PROJECT_ROOT

    return _build(
        context=pipeline.context,
        run_paths=pipeline._run_paths,
        run_id=pipeline._run_id,
        raw_context_output_dir=pipeline._raw_context_output_dir,
        project_root=PROJECT_ROOT,
    )


def resolve_stage_fragment_arg(pipeline: "MainPipeline") -> str:
    """Return an on-disk fragment/master path from dedup/merge/extract results ("" if none)."""
    from .stages.common import resolve_stage_fragment

    return resolve_stage_fragment(pipeline._stage_results)


def collect_stats(pipeline: "MainPipeline", result: "StageResult") -> None:
    """Accumulate stage statistics into the pipeline stats."""
    if hasattr(result, "llm_calls") and result.llm_calls:
        pipeline._stats.llm_calls += result.llm_calls
    if hasattr(result, "entity_count") and result.entity_count:
        pipeline._stats.entity_count += result.entity_count
    if hasattr(result, "relation_count") and result.relation_count:
        pipeline._stats.relation_count += result.relation_count
    if hasattr(result, "evidence_count") and result.evidence_count:
        pipeline._stats.evidence_count += result.evidence_count


def collect_output_files(pipeline: "MainPipeline", result: "StageResult") -> None:
    """Accumulate stage output files into the pipeline output list."""
    if hasattr(result, "output_files") and result.output_files:
        pipeline._output_files.extend(result.output_files)


def log_summary(pipeline: "MainPipeline", result: Any) -> None:
    """Print the run execution summary."""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 Pipeline Summary: {result.run_id}")
    logger.info(f"{'='*60}")
    logger.info(f"Status: {result.status.value}")
    logger.info(f"Stages: {result.stats.completed_stages}/{result.stats.total_stages} completed")
    logger.info(f"Latency: {result.stats.total_latency_ms:.1f}ms")
    logger.info(f"LLM Calls: {result.stats.llm_calls}")
    logger.info(f"Entities: {result.stats.entity_count}")
    logger.info(f"Relations: {result.stats.relation_count}")
    logger.info(f"Evidence: {result.stats.evidence_count}")

    if result.errors:
        logger.info(f"\n❌ Errors ({len(result.errors)}):")
        for error in result.errors[:5]:
            logger.info(f"   - {error}")

    if result.warnings:
        logger.info(f"\n⚠️ Warnings ({len(result.warnings)}):")
        for warning in result.warnings[:5]:
            logger.info(f"   - {warning}")

    logger.info(f"\n📁 Output files ({len(result.output_files)}):")
    for output_file in result.output_files[:10]:
        logger.info(f"   - {output_file}")

    logger.info(f"\n{'='*60}")
