"""Detailed diagnostics for regression artifacts and live master ontology.

The report distinguishes source provenance (input / injected / master_reused /
derived) and traces relation types. Diagnostic only — does not alter gates.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

JsonDict = Dict[str, Any]
PathLike = Union[str, Path]


def _entities_from_stage(stage: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = ((stage.get("metadata") or {}).get("stage_result_data") or {})
    return _entities_from_payload(data)


def _relations_from_stage(stage: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = ((stage.get("metadata") or {}).get("stage_result_data") or {})
    return _relations_from_payload(data)


def _entities_from_payload(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for key in ("nodes", "instances", "entities"):
        value = data.get(key)
        if isinstance(value, list):
            out.extend(x for x in value if isinstance(x, dict))
        elif isinstance(value, dict):
            for bucket in value.values():
                if isinstance(bucket, list):
                    out.extend(x for x in bucket if isinstance(x, dict))
    return out


def _relations_from_payload(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = data.get("relationships") or data.get("relations") or data.get("edges") or []
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def _source_kind(item: Dict[str, Any]) -> str:
    explicit = str(item.get("origin_kind") or "").strip()
    if explicit in {"input", "injected", "master_reused", "derived"}:
        return explicit
    if explicit:
        # Unknown explicit labels kept as-is for visibility.
        return explicit

    provenance = item.get("provenance") or {}
    attributes = item.get("attributes") or {}
    auto_generated = attributes.get("auto_generated") if isinstance(attributes, dict) else any(
        isinstance(a, dict) and a.get("name") == "auto_generated" and str(a.get("value")).lower() == "true"
        for a in (attributes if isinstance(attributes, list) else [])
    )
    if item.get("auto_generated") or auto_generated:
        return "injected"
    if isinstance(provenance, dict):
        origin = str(provenance.get("origin") or "").lower()
        if origin in {
            "deterministic_domain_entity",
            "deterministic_domain_relation",
            "deterministic_safety_net",
            "ecn_backbone",
            "deterministic_narrow_evidence",
        }:
            return "injected"
        if provenance.get("stage") == "extract":
            return "input"
    evidence = str(item.get("evidence") or item.get("evidence_ref") or "")
    if "backbone injection" in evidence.lower() or evidence.startswith("ECN:"):
        return "injected"
    if item.get("source_file"):
        return "input"
    return "master_or_unknown"


def _relation_type(rel: Dict[str, Any]) -> str:
    return str(rel.get("relation_type") or rel.get("type") or rel.get("rel_type") or "").upper()


def source_breakdown(stage: Dict[str, Any]) -> Dict[str, Any]:
    """Count entities and relations by available provenance signals."""
    entities = _entities_from_stage(stage)
    relations = _relations_from_stage(stage)
    return summarize_sources(entities, relations)


def summarize_sources(
    entities: Iterable[Dict[str, Any]],
    relations: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    entity_list = [e for e in entities if isinstance(e, dict)]
    relation_list = [r for r in relations if isinstance(r, dict)]
    entity_counts = Counter(_source_kind(x) for x in entity_list)
    relation_counts = Counter(_source_kind(x) for x in relation_list)
    type_counts = Counter(_relation_type(r) for r in relation_list if _relation_type(r))
    related_to = type_counts.get("RELATED_TO", 0)
    known_entity = ("input", "injected", "derived", "master_reused")
    known_rel = known_entity
    return {
        "entity_total": len(entity_list),
        "relation_total": len(relation_list),
        "entities_by_source": dict(entity_counts),
        "relations_by_source": dict(relation_counts),
        "relations_by_type": dict(type_counts),
        "related_to_count": related_to,
        "related_to_share": round(related_to / max(len(relation_list), 1), 4),
        "injected_entity_share": round(entity_counts.get("injected", 0) / max(len(entity_list), 1), 4),
        "injected_relation_share": round(relation_counts.get("injected", 0) / max(len(relation_list), 1), 4),
        "extracted_entity_share": round(entity_counts.get("input", 0) / max(len(entity_list), 1), 4),
        "extracted_relation_share": round(relation_counts.get("input", 0) / max(len(relation_list), 1), 4),
        "entity_source_coverage": round(
            sum(entity_counts[k] for k in known_entity) / max(len(entity_list), 1), 4
        ),
        "relation_source_coverage": round(
            sum(relation_counts[k] for k in known_rel) / max(len(relation_list), 1), 4
        ),
    }


def master_source_breakdown(
    master_dir: PathLike,
    *,
    nodes_name: str = "nodes.json",
    relationships_name: str = "relationships.json",
) -> Dict[str, Any]:
    """Breakdown for a materialized master_ontology directory."""
    root = Path(master_dir)
    nodes_path = root / nodes_name
    rels_path = root / relationships_name
    nodes_raw = json.loads(nodes_path.read_text(encoding="utf-8")) if nodes_path.exists() else []
    rels_raw = json.loads(rels_path.read_text(encoding="utf-8")) if rels_path.exists() else []
    if isinstance(nodes_raw, dict):
        entities = _entities_from_payload({"entities": nodes_raw, "nodes": nodes_raw.get("nodes")})
        if not entities and "nodes" not in nodes_raw:
            entities = _entities_from_payload(nodes_raw)
    else:
        entities = [x for x in nodes_raw if isinstance(x, dict)]
    if isinstance(rels_raw, dict):
        relations = _relations_from_payload(rels_raw)
    else:
        relations = [x for x in rels_raw if isinstance(x, dict)]
    summary = summarize_sources(entities, relations)
    summary["master_dir"] = str(root)
    summary["nodes_path"] = str(nodes_path)
    summary["relationships_path"] = str(rels_path)
    return summary


def trace_relation_type(
    extract_stage: Dict[str, Any],
    relationship_stage: Dict[str, Any],
    relation_type: str,
) -> Dict[str, Any]:
    """Trace a relation type from extraction into relationship governance."""
    wanted = relation_type.upper()
    extracted = [
        r
        for r in _relations_from_stage(extract_stage)
        if str(r.get("type") or r.get("relation_type") or "").upper() == wanted
    ]
    governed = ((relationship_stage.get("metadata") or {}).get("stage_result_data") or {})
    violations = [
        v
        for v in governed.get("violations") or []
        if str(v.get("relationship_type") or "").upper() == wanted
    ]
    return {
        "relation_type": wanted,
        "extract_count": len(extracted),
        "extract_relations": [
            {
                k: r.get(k)
                for k in (
                    "from_name",
                    "to_name",
                    "from",
                    "to",
                    "type",
                    "relation_type",
                    "evidence_ref",
                    "origin_kind",
                )
            }
            for r in extracted
        ],
        "governance_violation_count": len(violations),
        "governance_violations": violations[:20],
        "classification": (
            "missing_at_extract"
            if not extracted
            else "present_at_extract_review_governance_or_merge"
        ),
    }
