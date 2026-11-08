"""Skill-contract validation (ENTITY_CLASSIFICATION_ROADMAP T3.2).

Skills graduate from "prompt fragments" to "governance rule packs": each
skill may declare a three-layer ``contract`` block in its YAML —

    contract:
      classification:      # 分类契约
        required_layers: [L1, L2]          # layers that must be populated
        noise_patterns: ["^TODO.*"]        # entity names matching = violation
      entity:              # 实体契约
        required_entity_types:             # per-class minimums + fields
          Asset: {min_count: 1, required_fields: [meta_id]}
        rag_required_fields: [description, evidence_ref]   # RAG-index needs
      relation:            # 关系契约
        forbidden_relation_types: [DERIVED_FROM]
        evidence_required_relation_types: [DEPENDS_ON]

``validate_extraction_against_contract`` is a pure function over the legacy
task-oriented extraction dict; it never raises and returns structured
violations. Wiring point: the rule_l25 role of extraction_agents (skills are
enforced, not just suggested).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

__all__ = [
    "CONTRACT_CLASSIFICATION",
    "CONTRACT_ENTITY",
    "CONTRACT_RELATION",
    "validate_extraction_against_contract",
]

CONTRACT_CLASSIFICATION = "classification"
CONTRACT_ENTITY = "entity"
CONTRACT_RELATION = "relation"

_LAYER_BUCKETS = (
    ("L0_entities", "L0"),
    ("L1_entities", "L1"),
    ("L2_entities", "L2"),
    ("L2_5_entities", "L2.5"),
    ("L3_entities", "L3"),
)

_REL_TYPE_KEYS = ("type", "relation_type")


def _violation(layer: str, code: str, message: str, **extra: Any) -> Dict[str, Any]:
    out = {"contract_layer": layer, "code": code, "message": message}
    out.update(extra)
    return out


def _iter_entities(legacy: Dict[str, Any]):
    for bucket, _layer in _LAYER_BUCKETS:
        for raw in legacy.get(bucket) or []:
            if isinstance(raw, dict):
                yield bucket, raw


def validate_extraction_against_contract(
    legacy_extraction: Dict[str, Any],
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a legacy extraction dict against a skill contract. Pure.

    Returns::

        {
          "ok": bool,               # no error-severity violations
          "violations": [...],      # {contract_layer, code, message, ...}
          "stats": {...},           # per-layer counters for observability
        }
    """
    violations: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}

    # --- classification layer ------------------------------------------
    cls_cfg = contract.get(CONTRACT_CLASSIFICATION) or {}
    layer_counts: Dict[str, int] = {}
    for bucket, layer in _LAYER_BUCKETS:
        items = legacy_extraction.get(bucket) or []
        layer_counts[layer] = sum(1 for x in items if isinstance(x, dict))

    noise_patterns = cls_cfg.get("noise_patterns") or []
    compiled = []
    for pat in noise_patterns:
        try:
            compiled.append((pat, re.compile(pat)))
        except re.error:
            # invalid regex in skill config: report once, never raise
            violations.append(_violation(
                CONTRACT_CLASSIFICATION, "NOISE_PATTERN_INVALID",
                f"noise_patterns entry is not a valid regex: {pat!r}",
                pattern=str(pat),
            ))

    if compiled:
        noise_hits = 0
        for _bucket, raw in _iter_entities(legacy_extraction):
            name = str(raw.get("name") or raw.get("meta_name") or "")
            if any(rx.search(name) for _pat, rx in compiled):
                noise_hits += 1
        stats["noise_hits"] = noise_hits
        if noise_hits:
            violations.append(_violation(
                CONTRACT_CLASSIFICATION, "NOISE_PATTERN_HIT",
                f"{noise_hits} entity name(s) match declared noise patterns",
                count=noise_hits,
            ))

    for req_layer in cls_cfg.get("required_layers") or []:
        if layer_counts.get(req_layer, 0) == 0:
            violations.append(_violation(
                CONTRACT_CLASSIFICATION, "REQUIRED_LAYER_EMPTY",
                f"layer {req_layer!r} is declared required but has no entities",
                target_layer=req_layer,
            ))
    stats["layer_counts"] = layer_counts

    # --- entity layer ----------------------------------------------------
    ent_cfg = contract.get(CONTRACT_ENTITY) or {}
    rag_fields = [f for f in (ent_cfg.get("rag_required_fields") or [])]
    type_reqs = ent_cfg.get("required_entity_types") or {}

    counts_by_class: Dict[str, int] = {}
    entities_by_class: Dict[str, List[Dict[str, Any]]] = {}
    for _bucket, raw in _iter_entities(legacy_extraction):
        mclass = str(raw.get("meta_class") or raw.get("type") or "")
        counts_by_class[mclass] = counts_by_class.get(mclass, 0) + 1
        entities_by_class.setdefault(mclass, []).append(raw)

        missing_rag = [
            f for f in rag_fields
            if not str(raw.get(f) or "").strip()
        ]
        if missing_rag:
            violations.append(_violation(
                CONTRACT_ENTITY, "RAG_REQUIRED_FIELD_MISSING",
                f"entity {raw.get('meta_id') or raw.get('name')!r} lacks RAG-required field(s): {missing_rag}",
                entity_id=str(raw.get("meta_id") or ""),
                missing=missing_rag,
            ))

    for mclass, req in type_reqs.items():
        if not isinstance(req, dict):
            continue
        min_count = int(req.get("min_count") or 0)
        actual = counts_by_class.get(mclass, 0)
        if min_count and actual < min_count:
            violations.append(_violation(
                CONTRACT_ENTITY, "REQUIRED_ENTITY_TYPE_UNDERCOUNT",
                f"class {mclass!r} requires >= {min_count} entities, found {actual}",
                meta_class=mclass, min_count=min_count, actual=actual,
            ))
        for field in req.get("required_fields") or []:
            if not any(
                str(r.get(field) or "").strip()
                for r in entities_by_class.get(mclass, [])
            ):
                violations.append(_violation(
                    CONTRACT_ENTITY, "REQUIRED_FIELD_MISSING_FOR_CLASS",
                    f"no {mclass!r} entity fills required field {field!r}",
                    meta_class=mclass, field=field,
                ))
    stats["class_counts"] = counts_by_class

    # --- relation layer ---------------------------------------------------
    rel_cfg = contract.get(CONTRACT_RELATION) or {}
    forbidden = set(rel_cfg.get("forbidden_relation_types") or [])
    evidence_required = set(rel_cfg.get("evidence_required_relation_types") or [])
    rel_stats = {"total": 0, "forbidden_hits": 0, "evidence_missing": 0}
    for rel in legacy_extraction.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        rel_stats["total"] += 1
        rtype = ""
        for k in _REL_TYPE_KEYS:
            if str(rel.get(k) or "").strip():
                rtype = str(rel[k]).strip()
                break
        if rtype in forbidden:
            rel_stats["forbidden_hits"] += 1
            violations.append(_violation(
                CONTRACT_RELATION, "FORBIDDEN_RELATION_USED",
                f"relation type {rtype!r} is forbidden by this skill's contract",
                relation_type=rtype,
            ))
        if rtype in evidence_required and not str(rel.get("evidence_ref") or "").strip():
            rel_stats["evidence_missing"] += 1
            violations.append(_violation(
                CONTRACT_RELATION, "EVIDENCE_REQUIRED_MISSING",
                f"relation type {rtype!r} requires an evidence_ref",
                relation_type=rtype,
            ))
    stats["relations"] = rel_stats

    return {"ok": not violations, "violations": violations, "stats": stats}
