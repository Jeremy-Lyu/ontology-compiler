from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.infrastructure.services.entity_classification_rule_engine import (
    ClassificationResult,
    EntityClassificationRuleEngine,
)

ALLOWED_LAYERS = {"L0", "L1", "L2", "L2.5", "L3"}
ALLOWED_CLASSES = {"Capability", "System", "Actor", "Asset", "Data", "Process", "Concept", "UNKNOWN"}


@dataclass
class ClassificationCompareItem:
    entity_id: str
    name: str
    primary_meta_class: str
    primary_layer: str
    fallback_meta_class: str
    fallback_layer: str
    schema_valid: bool
    fallback_schema_valid: bool
    same_class: bool
    same_layer: bool
    primary_confidence: float
    fallback_confidence: float
    needs_review: bool
    fallback_decision: bool


@dataclass
class ClassificationQualityReport:
    summary: Dict[str, Any] = field(default_factory=dict)
    validity: Dict[str, Any] = field(default_factory=dict)
    distribution: Dict[str, Any] = field(default_factory=dict)
    comparisons: List[Dict[str, Any]] = field(default_factory=list)
    provider_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "validity": dict(self.validity),
            "distribution": dict(self.distribution),
            "comparisons": list(self.comparisons),
            "provider_info": dict(self.provider_info),
        }


@dataclass
class ClassificationSummary:
    total_entities: int = 0
    classified_entities: int = 0
    unknown_count: int = 0
    fallback_count: int = 0
    schema_violation_count: int = 0
    fallback_schema_violation_count: int = 0
    primary_class_distribution: Dict[str, int] = field(default_factory=dict)
    primary_layer_distribution: Dict[str, int] = field(default_factory=dict)
    fallback_class_distribution: Dict[str, int] = field(default_factory=dict)
    fallback_layer_distribution: Dict[str, int] = field(default_factory=dict)
    comparison_mismatch_count: int = 0
    unknown_rate: float = 0.0
    layer_distribution: Dict[str, int] = field(default_factory=dict)
    comparisons: List[ClassificationCompareItem] = field(default_factory=list)

    def to_report(self) -> ClassificationQualityReport:
        return ClassificationQualityReport(
            summary={
                "total_entities": self.total_entities,
                "classified_entities": self.classified_entities,
                "unknown_count": self.unknown_count,
                "unknown_rate": self.unknown_rate,
                "fallback_count": self.fallback_count,
            },
            validity={
                "schema_violation_count": self.schema_violation_count,
                "fallback_schema_violation_count": self.fallback_schema_violation_count,
                "comparison_mismatch_count": self.comparison_mismatch_count,
            },
            distribution={
                "primary_class_distribution": dict(self.primary_class_distribution),
                "primary_layer_distribution": dict(self.primary_layer_distribution),
                "fallback_class_distribution": dict(self.fallback_class_distribution),
                "fallback_layer_distribution": dict(self.fallback_layer_distribution),
                "layer_distribution": dict(self.layer_distribution),
            },
            comparisons=[item.__dict__ for item in self.comparisons],
            provider_info={"primary_provider": "EntityClassificationRuleEngine", "fallback_provider": "HeuristicClassificationFallbackProvider"},
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_report().to_dict()


class HeuristicClassificationFallbackProvider:
    def classify(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        name = str(entity.get("meta_name") or entity.get("name") or "").strip()
        meta_id = str(entity.get("meta_id") or entity.get("entity_id") or "").strip()
        meta_class = str(entity.get("meta_class") or "UNKNOWN").strip() or "UNKNOWN"
        layer = str(entity.get("layer") or "L0").strip() or "L0"
        entity_type = str(entity.get("entity_type") or meta_class or "Concept").strip() or "Concept"

        if meta_id.startswith("L3-") or layer == "L3":
            layer = "L3"
            entity_type = "Instance"
        elif meta_id.startswith("L25-") or meta_id.startswith("L2.5-"):
            layer = "L2.5"
        elif meta_id.startswith("L2-"):
            layer = "L2"
        elif meta_id.startswith("L1-"):
            layer = "L1"
        elif meta_id.startswith("L0-"):
            layer = "L0"
        elif "rule" in name.lower() or "policy" in name.lower():
            layer = "L2.5"
            meta_class = "Data"
            entity_type = "Data"
        elif any(token in name.lower() for token in ["system", "platform", "app", "service"]):
            layer = "L3"
            meta_class = "System"
            entity_type = "System"
        elif any(token in name.lower() for token in ["capability", "ability", "process"]):
            layer = "L1"
            meta_class = "Capability"
            entity_type = "Capability"
        elif not name:
            meta_class = "UNKNOWN"
            entity_type = "Concept"

        return {
            **entity,
            "meta_class": meta_class,
            "layer": layer,
            "entity_type": entity_type,
            "classification_confidence": float(entity.get("classification_confidence", 0.5)),
            "needs_review": True,
            "fallback_decision": True,
            "classification_source": "heuristic_fallback",
        }


class EntityClassificationService:
    def __init__(self, primary_provider: Optional[EntityClassificationRuleEngine] = None, fallback_provider: Optional[HeuristicClassificationFallbackProvider] = None):
        self.primary_provider = primary_provider or EntityClassificationRuleEngine()
        self.fallback_provider = fallback_provider or HeuristicClassificationFallbackProvider()

    def _schema_valid(self, entity: Dict[str, Any]) -> bool:
        return str(entity.get("layer") or "") in ALLOWED_LAYERS and str(entity.get("meta_class") or "") in ALLOWED_CLASSES

    def classify_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        return self.primary_provider.classify_with_distribution(entity)

    def summarize(self, entities: Iterable[Dict[str, Any]]) -> ClassificationSummary:
        summary = ClassificationSummary()
        primary_class_counts = Counter()
        primary_layer_counts = Counter()
        fallback_class_counts = Counter()
        fallback_layer_counts = Counter()
        layer_counts = Counter()

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            summary.total_entities += 1
            primary = self.primary_provider.classify_with_distribution(entity)
            fallback = self.fallback_provider.classify(entity)

            primary_schema_valid = self._schema_valid(primary)
            fallback_schema_valid = self._schema_valid(fallback)
            if not primary_schema_valid:
                summary.schema_violation_count += 1
            if not fallback_schema_valid:
                summary.fallback_schema_violation_count += 1

            primary_class = str(primary.get("meta_class") or "UNKNOWN")
            primary_layer = str(primary.get("layer") or "L0")
            fallback_class = str(fallback.get("meta_class") or "UNKNOWN")
            fallback_layer = str(fallback.get("layer") or "L0")

            primary_class_counts[primary_class] += 1
            primary_layer_counts[primary_layer] += 1
            fallback_class_counts[fallback_class] += 1
            fallback_layer_counts[fallback_layer] += 1
            layer_counts[primary_layer] += 1

            if primary_class == "UNKNOWN" or primary.get("fallback_decision"):
                summary.unknown_count += 1
            if primary.get("fallback_decision"):
                summary.fallback_count += 1
            if primary_class != fallback_class or primary_layer != fallback_layer:
                summary.comparison_mismatch_count += 1

            summary.comparisons.append(
                ClassificationCompareItem(
                    entity_id=str(entity.get("meta_id") or entity.get("entity_id") or entity.get("id") or ""),
                    name=str(entity.get("meta_name") or entity.get("name") or ""),
                    primary_meta_class=primary_class,
                    primary_layer=primary_layer,
                    fallback_meta_class=fallback_class,
                    fallback_layer=fallback_layer,
                    schema_valid=primary_schema_valid,
                    fallback_schema_valid=fallback_schema_valid,
                    same_class=primary_class == fallback_class,
                    same_layer=primary_layer == fallback_layer,
                    primary_confidence=float(primary.get("classification_confidence", 0.0) or 0.0),
                    fallback_confidence=float(fallback.get("classification_confidence", 0.0) or 0.0),
                    needs_review=bool(primary.get("needs_review", False)),
                    fallback_decision=bool(primary.get("fallback_decision", False)),
                )
            )

        summary.classified_entities = summary.total_entities - summary.unknown_count
        summary.unknown_rate = (summary.unknown_count / summary.total_entities) if summary.total_entities else 0.0
        summary.primary_class_distribution = dict(primary_class_counts)
        summary.primary_layer_distribution = dict(primary_layer_counts)
        summary.fallback_class_distribution = dict(fallback_class_counts)
        summary.fallback_layer_distribution = dict(fallback_layer_counts)
        summary.layer_distribution = dict(layer_counts)
        return summary

    def summarize_from_fragment(self, fragment_path: str | Path) -> ClassificationSummary:
        path = Path(fragment_path)
        if not path.exists():
            return ClassificationSummary()
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        entities = []
        for key in ("nodes", "instances"):
            value = data.get(key, [])
            if isinstance(value, list):
                entities.extend([item for item in value if isinstance(item, dict)])
        return self.summarize(entities)


def get_classification_service() -> EntityClassificationService:
    return EntityClassificationService()
