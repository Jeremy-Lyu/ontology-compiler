"""UNKNOWN Governance Service - 统计和告警未知类型占比。

实现P0_EXECUTION_PLAN.md任务5：给UNKNOWN建治理阈值。

功能：
- 统计实体meta_class、关系relation_type中的UNKNOWN占比
- 与阈值比对，超过阈值触发告警
- 生成治理报告，列出需要关注的未知类型
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VALID_META_CLASSES = {"Capability", "System", "Actor", "Asset", "Data", "Process"}

DEFAULT_THRESHOLDS = {
    "meta_class_unknown_ratio": 0.15,
    "relation_type_unknown_ratio": 0.20,
    "entity_layer_unknown_ratio": 0.10,
    "absolute_unknown_limit": 5,
}


@dataclass
class UnknownStats:
    """UNKNOWN governance statistics."""
    total_entities: int = 0
    unknown_meta_class_count: int = 0
    unknown_meta_class_types: List[str] = field(default_factory=list)
    total_relations: int = 0
    unknown_relation_type_count: int = 0
    unknown_relation_types: List[str] = field(default_factory=list)
    total_layers: int = 0
    unknown_layer_count: int = 0

    @property
    def meta_class_unknown_ratio(self) -> float:
        return self.unknown_meta_class_count / self.total_entities if self.total_entities > 0 else 0.0

    @property
    def relation_type_unknown_ratio(self) -> float:
        return self.unknown_relation_type_count / self.total_relations if self.total_relations > 0 else 0.0

    @property
    def layer_unknown_ratio(self) -> float:
        return self.unknown_layer_count / self.total_layers if self.total_layers > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entities": self.total_entities,
            "unknown_meta_class_count": self.unknown_meta_class_count,
            "unknown_meta_class_ratio": round(self.meta_class_unknown_ratio, 4),
            "unknown_meta_class_types": self.unknown_meta_class_types,
            "total_relations": self.total_relations,
            "unknown_relation_type_count": self.unknown_relation_type_count,
            "unknown_relation_type_ratio": round(self.relation_type_unknown_ratio, 4),
            "unknown_relation_types": self.unknown_relation_types,
            "total_layers": self.total_layers,
            "unknown_layer_count": self.unknown_layer_count,
            "layer_unknown_ratio": round(self.layer_unknown_ratio, 4),
        }


@dataclass
class UnknownAlert:
    """Alert triggered when UNKNOWN ratio exceeds threshold."""
    alert_type: str
    current_value: float
    threshold: float
    message: str
    severity: str  # "warning" or "critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "current_value": round(self.current_value, 4),
            "threshold": round(self.threshold, 4),
            "message": self.message,
            "severity": self.severity,
        }


class UnknownGovernanceService:
    """Service for tracking and alerting on UNKNOWN type ratios."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def analyze(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Tuple[UnknownStats, List[UnknownAlert]]:
        """Analyze entities and relationships for UNKNOWN types.

        Parameters
        ----------
        entities : list of dict
            Entity dicts with 'meta_class' and optionally 'layer' fields.
        relationships : list of dict
            Relationship dicts with 'relation_type' or 'type' field.

        Returns
        -------
        stats : UnknownStats
        alerts : list of UnknownAlert
        """
        stats = UnknownStats()
        alerts: List[UnknownAlert] = []

        unknown_mc_counter: Counter = Counter()
        unknown_rt_counter: Counter = Counter()

        for entity in entities:
            stats.total_entities += 1
            mc = str(entity.get("meta_class", "")).strip()
            if not mc or mc == "UNKNOWN" or mc not in VALID_META_CLASSES:
                stats.unknown_meta_class_count += 1
                unknown_mc_counter[mc or "(empty)"] += 1

            layer = str(entity.get("layer", "")).strip()
            stats.total_layers += 1
            if layer and layer not in {"L0", "L1", "L2", "L2.5", "L3"}:
                stats.unknown_layer_count += 1

        stats.unknown_meta_class_types = [f"{t}({c})" for t, c in unknown_mc_counter.most_common(10)]

        for rel in relationships:
            stats.total_relations += 1
            rt = str(rel.get("relation_type", "") or rel.get("type", "") or rel.get("relation", "")).strip()
            if not rt or rt == "UNKNOWN":
                stats.unknown_relation_type_count += 1
                unknown_rt_counter[rt or "(empty)"] += 1

        stats.unknown_relation_types = [f"{t}({c})" for t, c in unknown_rt_counter.most_common(10)]

        if stats.meta_class_unknown_ratio > self.thresholds["meta_class_unknown_ratio"]:
            severity = "critical" if stats.meta_class_unknown_ratio > self.thresholds["meta_class_unknown_ratio"] * 2 else "warning"
            alerts.append(UnknownAlert(
                alert_type="meta_class_unknown_ratio",
                current_value=stats.meta_class_unknown_ratio,
                threshold=self.thresholds["meta_class_unknown_ratio"],
                message=f"meta_class UNKNOWN ratio {stats.meta_class_unknown_ratio:.1%} exceeds threshold {self.thresholds['meta_class_unknown_ratio']:.1%}",
                severity=severity,
            ))

        if stats.relation_type_unknown_ratio > self.thresholds["relation_type_unknown_ratio"]:
            severity = "critical" if stats.relation_type_unknown_ratio > self.thresholds["relation_type_unknown_ratio"] * 2 else "warning"
            alerts.append(UnknownAlert(
                alert_type="relation_type_unknown_ratio",
                current_value=stats.relation_type_unknown_ratio,
                threshold=self.thresholds["relation_type_unknown_ratio"],
                message=f"relation_type UNKNOWN ratio {stats.relation_type_unknown_ratio:.1%} exceeds threshold {self.thresholds['relation_type_unknown_ratio']:.1%}",
                severity=severity,
            ))

        if stats.unknown_meta_class_count > self.thresholds["absolute_unknown_limit"]:
            alerts.append(UnknownAlert(
                alert_type="absolute_unknown_limit",
                current_value=float(stats.unknown_meta_class_count),
                threshold=self.thresholds["absolute_unknown_limit"],
                message=f"Absolute UNKNOWN count {stats.unknown_meta_class_count} exceeds limit {self.thresholds['absolute_unknown_limit']}",
                severity="warning",
            ))

        return stats, alerts

    def generate_report(self, stats: UnknownStats, alerts: List[UnknownAlert]) -> Dict[str, Any]:
        """Generate a governance report."""
        return {
            "summary": stats.to_dict(),
            "alerts": [a.to_dict() for a in alerts],
            "has_critical": any(a.severity == "critical" for a in alerts),
            "has_warning": any(a.severity == "warning" for a in alerts),
            "recommendation": (
                "CRITICAL: Immediate action required - UNKNOWN ratios critically high"
                if any(a.severity == "critical" for a in alerts)
                else "WARNING: Review UNKNOWN types and update prompts or mappings"
                if alerts
                else "OK: All UNKNOWN ratios within acceptable thresholds"
            ),
        }
