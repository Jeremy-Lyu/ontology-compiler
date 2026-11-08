"""Asset packet generation — extracted from MainPipeline (Phase 3).

Generates the standard asset package (entities / relations / evidence /
manifest / quality report / provenance) plus the retrieval asset packet.
These functions were moved out of `MainPipeline._generate_asset_packet` so the
pipeline orchestrator stays thin. They operate on a MainPipeline instance via a
small set of private collaborators; behavior is locked by
`tests/pipeline/test_asset_packet_characterization.py`.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.domain.models.core import PipelineExecutionResult
from .result_helpers import (
    deduplicate_entities,
    deduplicate_relations,
    extract_evidence_from_entities,
)
from .stages import StageResult

logger = logging.getLogger(__name__)


def collect_packet_data(pipeline, stage_results: Dict[str, StageResult]) -> Tuple[List[Any], List[Any], List[Any]]:
    """Collect entities/relations/evidence from stage data, fragment, and governed files."""
    entities, relations, evidence = [], [], []
    for stage_name, stage_result in stage_results.items():
        stage_data = stage_result.data or {}
        for key, value in stage_data.items():
            if isinstance(value, list) and key.endswith("_entities"):
                entities.extend(value)
            elif isinstance(value, list) and key == "nodes":
                entities.extend(value)
            elif isinstance(value, list) and (key == "relationships" or key == "relations"):
                relations.extend(value)
            elif isinstance(value, list) and key == "evidence":
                evidence.extend(value)

        fragment_path = stage_result.fragment_path or stage_data.get("fragment_path")
        if fragment_path:
            fragment_path = pipeline._normalize_artifact_path(fragment_path)
            if fragment_path == "." or not os.path.isfile(fragment_path):
                print(f"[AssetPacket] Skipping invalid fragment path for stage '{stage_name}': {fragment_path}")
            elif os.path.exists(fragment_path):
                try:
                    with open(fragment_path, 'r', encoding='utf-8') as f:
                        fragment_data = json.load(f)
                        if isinstance(fragment_data.get("nodes"), list):
                            entities.extend(fragment_data["nodes"])
                        if isinstance(fragment_data.get("relationships"), list):
                            relations.extend(fragment_data["relationships"])
                        if isinstance(fragment_data.get("evidence"), list):
                            evidence.extend(fragment_data["evidence"])
                except Exception as e:
                    print(f"[AssetPacket] Error reading fragment {fragment_path}: {e}")
        governed_fragment_path = stage_data.get("governed_fragment_path")
        if governed_fragment_path:
            governed_fragment_path = pipeline._normalize_artifact_path(governed_fragment_path)
            if governed_fragment_path != "." and os.path.isfile(governed_fragment_path):
                try:
                    with open(governed_fragment_path, 'r', encoding='utf-8') as f:
                        governed_data = json.load(f)
                        if isinstance(governed_data.get("nodes"), list):
                            entities.extend(governed_data["nodes"])
                        if isinstance(governed_data.get("relationships"), list):
                            relations.extend(governed_data["relationships"])
                        if isinstance(governed_data.get("evidence"), list):
                            evidence.extend(governed_data["evidence"])
                except Exception as e:
                    print(f"[AssetPacket] Error reading governed fragment {governed_fragment_path}: {e}")
    return entities, relations, evidence


def prepare_packet_entities(pipeline, entities: List[Any], relations: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Attach provenance and deduplicate packet entities + relations."""
    entities = pipeline._asset_packet_service.add_provenance_to_entities(
        entities,
        source_file=pipeline.context.source_path,
        run_id=pipeline._run_id,
        rule_version=pipeline.context.metadata.get("rule_version", "1.0"),
        model_version=pipeline.context.metadata.get("model_version", "deepseek-chat"),
        skill_version=pipeline.context.metadata.get("skill_version", "2.0.0"),
    )
    entities = deduplicate_entities(entities)

    relations = pipeline._asset_packet_service.add_provenance_to_relations(
        relations,
        source_file=pipeline.context.source_path,
        run_id=pipeline._run_id,
        rule_version=pipeline.context.metadata.get("rule_version", "1.0"),
        model_version=pipeline.context.metadata.get("model_version", "deepseek-chat"),
        skill_version=pipeline.context.metadata.get("skill_version", "2.0.0"),
    )
    relations = deduplicate_relations(relations)
    return entities, relations


def build_packet_metadata(pipeline, quality_score: float, governed: bool) -> Dict[str, Any]:
    """Build the asset-packet metadata payload."""
    return {
        "input_files": [pipeline.context.source_path],
        "pipeline_version": "1.2.0",
        "rule_version": pipeline.context.metadata.get("rule_version", "2.1.0"),
        "model_version": pipeline.context.metadata.get("model_version", "deepseek-chat"),
        "skill_versions": {"entity_recognition": "2.0.0", "relation_extraction": "2.0.0"},
        "prompt_version": "3.0.0",
        "execution_duration_ms": int(pipeline._stats.total_latency_ms),
        "llm_call_count": pipeline._stats.llm_calls,
        "quality_score": quality_score,
        "review_required": quality_score < 0.7,
        "errors": pipeline._errors[:5],
        "warnings": pipeline._warnings[:5],
    }


def build_entity_refs(pipeline, entities: List[Any]) -> List[Dict[str, Any]]:
    """Build retrieval-packet entity references."""
    return [
        {
            "entity_id": str(entity.get("meta_id") or entity.get("entity_id") or entity.get("id") or ""),
            "canonical_name": str(entity.get("name") or entity.get("canonical_name") or ""),
            "aliases": entity.get("aliases", []) if isinstance(entity.get("aliases"), list) else [],
            "entity_type": str(entity.get("meta_class") or entity.get("type") or ""),
            "layer": str(entity.get("layer") or ""),
            "confidence": float(entity.get("confidence", 0.0) or 0.0),
            "validation_state": str(entity.get("validation_state") or "pending"),
            "metadata": {"source_file": entity.get("source_file", pipeline.context.source_path)},
        }
        for entity in entities
        if isinstance(entity, dict)
    ]


def build_relation_refs(relations: List[Any]) -> List[Dict[str, Any]]:
    """Build retrieval-packet relation references."""
    return [
        {
            "relation_id": str(relation.get("relation_id") or relation.get("id") or ""),
            "source_id": str(relation.get("source") or relation.get("source_id") or relation.get("from") or ""),
            "target_id": str(relation.get("target") or relation.get("target_id") or relation.get("to") or ""),
            "relation_type": str(relation.get("type") or relation.get("relation_type") or ""),
            "confidence": float(relation.get("confidence", 0.0) or 0.0),
            "evidence_ids": relation.get("evidence_ids", []) if isinstance(relation.get("evidence_ids"), list) else [],
            "validation_state": str(relation.get("validation_state") or "pending"),
        }
        for relation in relations
        if isinstance(relation, dict)
    ]


def build_evidence_refs(pipeline, evidence: List[Any]) -> List[Dict[str, Any]]:
    """Build retrieval-packet evidence references."""
    return [
        {
            "evidence_id": str(item.get("id") or item.get("evidence_id") or ""),
            "source_document_id": str(item.get("source_file") or pipeline.context.source_path),
            "source_chunk_id": str(item.get("chunk_id") or ""),
            "span": str(item.get("content") or item.get("raw_text") or item.get("context") or ""),
            "support_level": "direct" if item.get("content") or item.get("raw_text") else "indirect",
            "confidence": float(item.get("confidence", 0.0) or 0.0),
        }
        for item in evidence
        if isinstance(item, dict)
    ]


def build_retrieval_packet(
    pipeline,
    package_path: str,
    entities: List[Any],
    relations: List[Any],
    evidence: List[Any],
    quality_score: float,
    governed: bool,
) -> Tuple[str, Any]:
    """Build and save the retrieval asset packet; returns (packet_path, packet)."""
    document_path = Path(pipeline.context.source_path)
    return pipeline._retrieval_asset_packet_service.build_and_save_packet(
        run_id=pipeline._run_id,
        ontology_version=str(pipeline.context.metadata.get("ontology_version", "1.0.0")),
        trace_context=pipeline._build_trace_context(artifact_id=f"retrieval-{pipeline._run_id}"),
        document_ref={
            "document_id": document_path.name,
            "document_type": pipeline.context.metadata.get("document_type", ""),
            "source_path": pipeline.context.source_path,
            "hash": pipeline.context.metadata.get("source_hash", ""),
            "title": document_path.stem,
        },
        chunk_ref={
            "chunk_id": f"run-{pipeline._run_id}",
            "document_id": document_path.name,
            "evidence_span": "",
            "metadata": {"package_path": package_path},
        },
        entity_refs=build_entity_refs(pipeline, entities),
        relation_refs=build_relation_refs(relations),
        evidence_refs=build_evidence_refs(pipeline, evidence),
        structured_filters={"domain": pipeline.context.domain},
        retrieval_text=[
            str(entity.get("name"))
            for entity in entities
            if isinstance(entity, dict) and entity.get("name")
        ][:500],
        quality={
            "confidence": quality_score,
            "evidence_coverage": round(len(evidence) / max(len(entities), 1), 4),
            "entity_count": len(entities),
            "relation_count": len(relations),
        },
        provenance={
            "pipeline_run_id": pipeline._run_id,
            "package_path": package_path,
            "source_path": pipeline.context.source_path,
        },
    )


def generate_asset_packet(pipeline, result: PipelineExecutionResult) -> None:
    """生成标准资产包（Phase C - Asset Packaging）

    将 pipeline 产物转换为标准资产包格式，包含：
    - entities.json: 实体数据（含 provenance）
    - relations.json: 关系数据（含 provenance）
    - evidence.json: 证据片段
    - manifest.json: 资产清单
    - quality_report.json: 质量报告
    - provenance.json: 溯源记录
    """
    try:
        entities, relations, evidence = collect_packet_data(pipeline, pipeline._stage_results)
        entities, relations = prepare_packet_entities(pipeline, entities, relations)
        evidence = extract_evidence_from_entities(entities)

        quality_score = pipeline._stage_results.get("audit", StageResult("", False, "", {})).data.get("quality_score", 0.0)
        governed = quality_score >= 0.7
        metadata = build_packet_metadata(pipeline, quality_score, governed)

        package_path = pipeline._asset_packet_service.generate_asset_packet(
            run_id=pipeline._run_id,
            session_id=pipeline._session_id,
            entities=entities,
            relations=relations,
            evidence=evidence,
            metadata=metadata,
            governed=governed,
        )

        package_data = pipeline._asset_packet_service.load_asset_packet(pipeline._run_id, governed=governed)
        if package_data:
            pipeline._asset_index_service.index_package(pipeline._run_id, package_data, package_path, governed)

        result.output_files.append(package_path)
        packet_path, retrieval_packet = build_retrieval_packet(
            pipeline, package_path, entities, relations, evidence, quality_score, governed
        )
        result.output_files.append(str(packet_path))
        pipeline.context.metadata["retrieval_asset_packet_path"] = str(packet_path)
        logger.info(f"📦 Generated asset package: {package_path} (governed={governed})")
        logger.info(f"🔎 Generated retrieval asset packet: {packet_path}")

        result.stats.entity_count = len(entities)
        result.stats.relation_count = len(relations)
        result.stats.evidence_count = len(evidence)
    except Exception as e:
        logger.warning(f"Failed to generate asset package: {e}")