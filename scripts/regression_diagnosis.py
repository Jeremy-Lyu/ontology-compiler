"""Regression-diagnosis helpers for comparing pipeline outputs.

These helpers keep diagnosis separate from acceptance policy. They identify
signals of master-ontology dominance and summarize golden failures without
changing the production runner's pass/fail thresholds.
"""
from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["summarize_regression_failures", "master_dominance_signal"]


def summarize_regression_failures(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return compact actionable failure rows from a regression summary."""
    rows: List[Dict[str, Any]] = []
    for result in summary.get("results") or []:
        evaluation = result.get("evaluation") or {}
        golden = evaluation.get("golden") or {}
        rows.append({
            "sample_id": result.get("sample_id", ""),
            "pass_fail": result.get("pass_fail", ""),
            "failed_checks": [c.get("name") for c in evaluation.get("checks") or [] if not c.get("passed")],
            "entity_count": (evaluation.get("observed") or {}).get("entity_count", 0),
            "relation_count": (evaluation.get("observed") or {}).get("relation_count", 0),
            "entity_recall": golden.get("entity_recall"),
            "relation_recall": golden.get("relation_recall"),
            "unmatched_entities": golden.get("unmatched_entities") or [],
            "unmatched_relations": golden.get("unmatched_relations") or [],
        })
    return rows


def master_dominance_signal(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Flag suspiciously stable output counts across heterogeneous samples."""
    rows = summarize_regression_failures(summary)
    entities = [r["entity_count"] for r in rows if r["entity_count"]]
    relations = [r["relation_count"] for r in rows if r["relation_count"]]
    entity_span = max(entities) - min(entities) if entities else 0
    relation_span = max(relations) - min(relations) if relations else 0
    return {
        "sample_count": len(rows),
        "entity_counts": entities,
        "relation_counts": relations,
        "entity_span": entity_span,
        "relation_span": relation_span,
        "suspiciously_stable": len(rows) >= 3 and entity_span <= 20 and relation_span <= 40,
        "reason": "Output counts vary little across samples; separate input contribution from master reuse." if len(rows) >= 3 and entity_span <= 20 and relation_span <= 40 else "No stability signal.",
    }
