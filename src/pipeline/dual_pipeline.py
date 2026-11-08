"""Legacy dual-pipeline compatibility layer.

This module preserves the historical import surface used by older tests and
migration tooling. The canonical production pipeline lives in
`src.pipeline.main_pipeline`. It is retained only for compatibility and
should not be used for new execution paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class PipelineMode(Enum):
    LEGACY = "legacy"
    NEW = "new"
    DUAL = "dual"


class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    RUNNING = "running"
    PENDING = "pending"


@dataclass
class PipelineResult:
    mode: PipelineMode
    status: ExecutionStatus
    entities: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    llm_calls: int = 0
    entity_count: int = 0
    relation_count: int = 0
    evidence_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "status": self.status.value,
            "entities": self.entities,
            "relations": self.relations,
            "evidence": self.evidence,
            "errors": self.errors,
            "latency_ms": self.latency_ms,
            "llm_calls": self.llm_calls,
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "evidence_count": self.evidence_count,
        }


@dataclass
class MigrationComparison:
    legacy_result: PipelineResult
    new_result: PipelineResult
    entity_overlap: float = 0.0
    entity_precision: float = 0.0
    entity_recall: float = 0.0


@dataclass
class MigrationReport:
    run_id: str
    timestamp: datetime
    comparison: MigrationComparison
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class UnifiedExtractorAdapter:
    @staticmethod
    def adapt_to_packet_format(legacy_result: Dict[str, Any]) -> Dict[str, Any]:
        entities: Dict[str, List[Dict[str, Any]]] = {}
        for item in legacy_result.get("nodes", []):
            layer = item.get("layer", "L0")
            entities.setdefault(layer, []).append(item)
        for item in legacy_result.get("instances", []):
            layer = item.get("layer", "L3")
            entities.setdefault(layer, []).append(item)

        relations = []
        for rel in legacy_result.get("relationships", []):
            relations.append(
                {
                    "source_id": rel.get("source", ""),
                    "target_id": rel.get("target", ""),
                    "relation_type": rel.get("type", ""),
                    "confidence": rel.get("confidence", 0.0),
                }
            )

        return {"entities": entities, "relations": relations, "evidence": legacy_result.get("evidence", [])}

    @staticmethod
    def adapt_to_legacy_format(new_result: Dict[str, Any]) -> Dict[str, Any]:
        nodes = []
        instances = []
        for layer, items in new_result.get("entities", {}).items():
            for item in items:
                item = dict(item)
                item.setdefault("layer", layer)
                if layer == "L3":
                    instances.append(item)
                else:
                    nodes.append(item)

        relationships = []
        for rel in new_result.get("relations", []):
            relationships.append(
                {
                    "source": rel.get("source_id", ""),
                    "target": rel.get("target_id", ""),
                    "type": rel.get("relation_type", ""),
                }
            )

        return {"nodes": nodes, "instances": instances, "relationships": relationships}


class DualPipeline:
    def __init__(self, mode: PipelineMode = PipelineMode.DUAL):
        self.mode = mode

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DualPipeline":
        mode = PipelineMode(config.get("mode", "dual"))
        return cls(mode=mode)


__all__ = [
    "PipelineMode",
    "ExecutionStatus",
    "PipelineResult",
    "MigrationComparison",
    "MigrationReport",
    "UnifiedExtractorAdapter",
    "DualPipeline",
]
