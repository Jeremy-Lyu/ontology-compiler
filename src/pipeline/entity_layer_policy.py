"""Entity-layer and profile policies (G7, B-7, 2026-08-17).

Extracted from ``MainPipeline`` — B-series removed orchestration side-effects;
this batch moves the *business decisions* (layer inference, profile summary,
metadata projection, layer-pair gating) into pure, testable functions that can
be reused by any orchestrator (e.g. skill cores). Interface with pipeline state
is injected (``lookup_layer`` / ``governance_engine``) rather than touched
directly.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .stages.common import _entity_layer_value  # noqa: F401


def infer_entity_layer(entity: Dict[str, Any]) -> Dict[str, Any]:
    """Infer a layer for an entity that has none (heuristics, no state access)."""
    if not isinstance(entity, dict):
        return entity
    if str(entity.get("layer", "")).strip():
        return entity
    name = str(entity.get("name", "")).lower()
    meta_class = str(entity.get("meta_class", entity.get("type", ""))).lower()
    evidence_blob = " ".join(
        str(entity.get(key, ""))
        for key in ("source_context", "evidence", "evidence_span", "description", "summary")
    ).lower()
    layer = None
    instance_tokens = ("benchmark", "example", "case", "site", "plant", "instance", "variant", "occurrence", "local")
    bridge_tokens = ("interface", "bridge", "control", "report", "inventory", "analysis", "mapping", "kpi")
    abstract_tokens = ("concept", "capability", "function", "policy", "method", "framework", "principle", "guideline")
    instance_hit = any(token in name for token in instance_tokens) or any(token in evidence_blob for token in instance_tokens)
    bridge_hit = any(token in meta_class for token in bridge_tokens) or any(token in name for token in bridge_tokens)
    abstract_hit = any(token in name for token in abstract_tokens) or any(token in meta_class for token in abstract_tokens)
    if instance_hit and not abstract_hit:
        layer = "L3"
    elif bridge_hit:
        layer = "L2.5"
    elif str(entity.get("type", "")).strip():
        layer = "L2"
    if layer:
        entity = dict(entity)
        entity["layer"] = layer
    return entity


def normalize_entity_record(entity: Any, normalize_fn: Any = None) -> Any:
    """Normalize then infer layer. ``normalize_fn`` injected (default no-op)."""
    if not isinstance(entity, dict):
        return entity
    if normalize_fn is not None:
        entity = normalize_fn(entity)
    return infer_entity_layer(entity)


def build_entity_layer_index(stage_data: Dict[str, Any]) -> Dict[str, str]:
    """Map entity id -> layer across nodes/instances/entities buckets."""
    id_to_layer: dict[str, str] = {}
    for key in ("nodes", "instances", "entities"):
        bucket = stage_data.get(key)
        if not isinstance(bucket, list):
            continue
        for ent in bucket:
            if not isinstance(ent, dict):
                continue
            eid = str(ent.get("meta_id") or ent.get("entity_id") or ent.get("id") or "")
            if not eid:
                continue
            id_to_layer[eid] = _entity_layer_value(ent)
    return id_to_layer


def is_layer_pair_blocked(
    rel: Dict[str, Any],
    *,
    lookup_layer: Any = None,
    governance_engine: Any = None,
) -> bool:
    """Check whether a relationship's source→target layer pair is blocked.

    Pure apart from two injected dependencies (G7):
    - ``lookup_layer(entity_id) -> str`` resolves layers by id (from pipeline
      stage results in production; from the index map in tests / other callers)
    - ``governance_engine.is_layer_pair_allowed(source, target)`` decides.

    Mirrors the legacy ``MainPipeline._is_layer_pair_blocked`` semantics.
    """
    source_layer = str(rel.get("source_layer") or rel.get("from_layer") or "").strip()
    target_layer = str(rel.get("target_layer") or rel.get("to_layer") or "").strip()

    if not source_layer or not target_layer:
        source_id = str(rel.get("source") or rel.get("from") or rel.get("source_id") or rel.get("from_id") or "")
        target_id = str(rel.get("target") or rel.get("to") or rel.get("target_id") or rel.get("to_id") or "")
        if lookup_layer is not None:
            if source_id and not source_layer:
                source_layer = str(lookup_layer(source_id)).strip() or ""
            if target_id and not target_layer:
                target_layer = str(lookup_layer(target_id)).strip() or ""

    if not source_layer or not target_layer:
        return False
    try:
        if governance_engine is None:
            from ontology_compiler.services.relation_governance import get_relation_governance_engine
            governance_engine = get_relation_governance_engine()
        return not governance_engine.is_layer_pair_allowed(source_layer, target_layer)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Entity profile summary
# ---------------------------------------------------------------------------


def init_entity_profile_summary(entity_count: int) -> dict[str, Any]:
    return {
        "count": entity_count,
        "canonical_names": [],
        "types": {},
        "layers": {},
        "with_aliases": 0,
        "with_normalized_from": 0,
        "with_provenance": 0,
        "with_evidence": 0,
        "with_structured_metadata": 0,
        "with_confidence": 0,
        "with_validation_state": 0,
        "profile_completeness": 0.0,
    }


def accumulate_entity_profile(entity: dict[str, Any], summary: dict[str, Any], seen_names: set) -> None:
    name = str(entity.get("name", "")).strip()
    if name and name not in seen_names:
        seen_names.add(name)
        summary["canonical_names"].append(name)
    etype = str(entity.get("type", "")).strip()
    layer = str(entity.get("layer", "")).strip()
    if etype:
        summary["types"][etype] = summary["types"].get(etype, 0) + 1
    if layer:
        summary["layers"][layer] = summary["layers"].get(layer, 0) + 1
    if entity.get("aliases"):
        summary["with_aliases"] += 1
    if entity.get("normalized_from"):
        summary["with_normalized_from"] += 1
    if entity.get("provenance") or entity.get("source_file") or entity.get("source_context"):
        summary["with_provenance"] += 1
    if entity.get("evidence") or entity.get("evidence_span"):
        summary["with_evidence"] += 1
    if entity.get("structured_metadata"):
        summary["with_structured_metadata"] += 1
    if entity.get("confidence") is not None:
        summary["with_confidence"] += 1
    if entity.get("validation_state"):
        summary["with_validation_state"] += 1


def count_entity_profile_completeness(entity: dict[str, Any]) -> int:
    return sum(
        1
        for key in ("aliases", "normalized_from", "provenance", "evidence_span", "structured_metadata", "confidence", "validation_state")
        if entity.get(key)
    )


def build_entity_profile_summary(entities: list[dict[str, Any]]) -> dict[str, Any]:
    summary = init_entity_profile_summary(len(entities))
    seen_names: set = set()
    completeness_total = 0
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        accumulate_entity_profile(entity, summary, seen_names)
        completeness_total += count_entity_profile_completeness(entity)
    summary["canonical_names"] = summary["canonical_names"][:50]
    summary["profile_completeness"] = round(completeness_total / max(1, len(entities)), 2)
    return summary


# ---------------------------------------------------------------------------
# Structured-metadata projection + domain context
# ---------------------------------------------------------------------------


def infer_domain_metadata_context(metadata: dict[str, Any]) -> dict[str, Any]:
    hints = metadata.get("source_hints") or []
    if isinstance(hints, str):
        hints = [hints]
    hint_text = " ".join(str(h).lower() for h in hints)
    context: dict[str, Any] = {}
    if any(token in hint_text for token in ("inventory", "plant", "supply", "wip", "goods")):
        context["operational_context"] = "supply_chain_inventory"
    if any(token in hint_text for token in ("mbom", "bom", "bop", "variant", "occurrence", "position")):
        context["manufacturing_context"] = "bom_centric"
    if any(token in hint_text for token in ("kpi", "benchmark", "report")):
        context["reporting_context"] = "kpi_benchmark"
    if any(token in hint_text for token in ("policy", "security", "compliance", "restricted", "proprietary")):
        context["policy_context"] = "compliance"
    return context


def project_structured_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    projected: dict[str, Any] = {}
    for key in ("effective_dates", "security_classes", "frequencies", "conditions", "formulas", "source_hints", "time_windows", "percentages", "ranges"):
        value = metadata.get(key)
        if value:
            projected[key] = value
    rewrite_context = metadata.get("rewrite_context")
    if isinstance(rewrite_context, dict) and rewrite_context:
        projected["rewrite_context"] = rewrite_context
    ontology_hint = metadata.get("ontology_hint")
    if ontology_hint:
        projected["ontology_hint"] = ontology_hint
    domain_context = infer_domain_metadata_context(projected)
    if domain_context:
        projected["domain_context"] = domain_context
    return projected