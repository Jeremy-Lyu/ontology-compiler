"""Adapter: task-oriented extraction JSON → OntologyPacket structure.

Phase 1 of the ENTITY_CLASSIFICATION_ROADMAP T1.2 (protocol adoption prep):
the extractor currently emits the legacy task-oriented shape
(``L0_entities/L1_entities/.../relationships`` with ``from/to/type``), while
the canonical output contract is ``schemas/ontology_packet.schema.json``
(document/layer_scope/entities/relations/evidence/provenance/quality/
warnings/assumptions).

This adapter maps legacy → packet **purely and deterministically** so both
shapes can coexist behind a feature flag without touching extraction
behavior. Field names follow
``docs/ENTITY_CLASSIFICATION_SPEC.md`` §3; the packet shell is §6.

Layer buckets map as: L0_entities/L1_entities/L2_entities/L2_5_entities/
L3_entities → entities[].layer = L0/L1/L2/L2.5/L3. Relationships use
``from/to`` entity names when ids are absent; unresolved endpoints are
reported in ``warnings``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

__all__ = [
    "adapt_extraction_json_to_packet",
    "iter_packet_entities",
    "maybe_attach_ontology_packet",
]

_LAYER_BUCKETS = (
    ("L0_entities", "L0"),
    ("L1_entities", "L1"),
    ("L2_entities", "L2"),
    ("L2_5_entities", "L2.5"),
    ("L3_entities", "L3"),
)


def _first(*values: Any) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    if values and isinstance(values[-1], str):
        return values[-1].strip()
    return ""


# T2.2 temporal discipline (docs/ENTITY_CLASSIFICATION_SPEC.md §5): T1/T2/T3
# business valid-time axes stay three independent columns — never merged.
# Aliases cover raw LLM output keys in EN/CN manufacturing documents.
_TEMPORAL_ALIASES = {
    "t1_planned_switch_date": (
        "t1", "t1_planned_switch_date", "planned_switch_date",
        "计划切换日", "计划切换日期",
    ),
    "t2_supplier_delivery_date": (
        "t2", "t2_supplier_delivery_date", "supplier_delivery_date",
        "供应商供货日", "供货日期",
    ),
    "t3_actual_install_date": (
        "t3", "t3_actual_install_date", "actual_install_date",
        "实际装车日", "实际装车日期", "装车日期",
    ),
    "valid_from": (
        "valid_from", "effective_from", "effective_date",
        "生效日期", "生效日",
    ),
    "valid_to": (
        "valid_to", "effective_to", "失效日期",
    ),
}


def _norm_key(key: Any) -> str:
    return str(key).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _extract_temporal_fields(raw: Dict[str, Any], attributes: Dict[str, Any]) -> Dict[str, str]:
    """Pull T1/T2/T3 + version interval out of top-level fields or attributes."""
    sources: List[tuple[str, Any]] = list(raw.items()) + list(attributes.items())
    found: Dict[str, str] = {}
    for canonical, aliases in _TEMPORAL_ALIASES.items():
        if found.get(canonical):
            continue
        for src_key, src_val in sources:
            nk = _norm_key(src_key)
            if not nk or not isinstance(src_val, str) or not src_val.strip():
                continue
            for alias in aliases:
                na = _norm_key(alias)
                if nk == na or (len(na) >= 4 and na in nk):
                    found[canonical] = src_val.strip()
                    break
            if found.get(canonical):
                break
    return found


def _adapt_entity(raw: Dict[str, Any], layer: str) -> Dict[str, Any]:
    attributes = raw.get("attributes")
    if not isinstance(attributes, dict):
        # legacy lists [{"name","value"}] or strings — normalize to dict
        norm: Dict[str, Any] = {}
        if isinstance(attributes, list):
            for item in attributes:
                if isinstance(item, dict) and item.get("name"):
                    norm[str(item["name"])] = item.get("value")
        elif isinstance(attributes, str) and attributes.strip():
            norm["note"] = attributes.strip()
        attributes = norm

    meta_class = _first(raw.get("meta_class"), raw.get("type"))
    parent = _first(raw.get("parent_id"), raw.get("parent_class"))
    temporal = _extract_temporal_fields(raw, attributes)

    return {
        "id": _first(raw.get("meta_id"), raw.get("id")),
        "code": _first(raw.get("code")),
        "name": _first(raw.get("name"), raw.get("meta_name")),
        "layer": layer,
        "layer_scope": _first(raw.get("layer_scope"), "spec"),
        "type": meta_class,
        "sub_type": _first(raw.get("meta_subclass"), raw.get("sub_type")),
        **({"status": st} if (st := _first(raw.get("status"))) else {}),
        "parent_id": parent,
        "description": _first(raw.get("description")),
        "domain": _first(raw.get("domain"), "ENTERPRISE"),
        "source_file": _first(raw.get("source_file")),
        "source_span": _first(raw.get("source_span")),
        "evidence_ref": _first(
            raw.get("evidence_ref"), raw.get("evidence")
        ),
        "version": _first(raw.get("version"), "1.0"),
        **({"confidence": raw["confidence"]}
           if isinstance(raw.get("confidence"), (int, float)) else {}),
        "attributes": attributes,
        "aliases": raw.get("aliases") if isinstance(raw.get("aliases"), list) else [],
        "tags": raw.get("tags") if isinstance(raw.get("tags"), list) else [],
        "relations_summary": {},
        **temporal,
    }


def adapt_extraction_json_to_packet(
    legacy: Dict[str, Any],
    *,
    document_name: str = "",
    domain: str = "",
) -> Dict[str, Any]:
    """Map a legacy extraction dict to the OntologyPacket shape.

    Pure function: *legacy* is never mutated. Relationship endpoints given as
    entity names are resolved against extracted ids; unresolved names are kept
    verbatim and reported in ``warnings``.
    """
    warnings: List[str] = []
    entities: List[Dict[str, Any]] = []

    name_to_id: Dict[str, str] = {}
    for bucket, layer in _LAYER_BUCKETS:
        for raw in legacy.get(bucket) or []:
            if not isinstance(raw, dict):
                continue
            ent = _adapt_entity(raw, layer)
            entities.append(ent)
            if ent["name"]:
                name_to_id.setdefault(ent["name"], ent["id"])

    id_set = {e["id"] for e in entities if e["id"]}
    relations: List[Dict[str, Any]] = []
    seen_rels = set()
    for rel in legacy.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        src_raw = rel.get("source_id") or rel.get("from") or ""
        tgt_raw = rel.get("target_id") or rel.get("to") or ""
        src = src_raw if src_raw in id_set else name_to_id.get(str(src_raw).strip(), str(src_raw).strip())
        tgt = tgt_raw if tgt_raw in id_set else name_to_id.get(str(tgt_raw).strip(), str(tgt_raw).strip())
        if not src or not tgt:
            continue
        rtype = _first(rel.get("type"), rel.get("relation_type"))
        key = (src, tgt, rtype)
        if key in seen_rels:
            continue
        seen_rels.add(key)
        rel_conf = rel.get("confidence")
        prov = _first(rel.get("provenance"), "llm")
        if prov not in ("llm", "skill", "rule", "inferred"):
            prov = "llm"  # schema enum: unknown legacy values fall back
        relations.append({
            "source_id": src,
            "target_id": tgt,
            "relation_type": rtype,
            **({"source_cq": cq} if (cq := _first(
                rel.get("source_cq"), rel.get("cq_id"), rel.get("来源CQ"))) else {}),
            "layer_scope": _first(rel.get("layer_scope"), "spec"),
            **({"confidence": rel_conf} if isinstance(rel_conf, (int, float)) else {}),
            "evidence_ref": _first(rel.get("evidence_ref")),
            "effective_from": _first(rel.get("effective_from")),
            "effective_to": _first(rel.get("effective_to")),
            "provenance": prov,
            "description": _first(rel.get("description")),
            "attributes": rel.get("attributes") if isinstance(rel.get("attributes"), dict) else {},
        })
        if src not in id_set:
            warnings.append({
                "code": "ENDPOINT_UNRESOLVED",
                "message": f"relation source endpoint not resolved: {src!r}",
                "severity": "warning", "relation_key": f"{src}->{rtype}",
            })
        if tgt not in id_set:
            warnings.append({
                "code": "ENDPOINT_UNRESOLVED",
                "message": f"relation target endpoint not resolved: {tgt!r}",
                "severity": "warning", "relation_key": f"->{tgt}",
            })

    file_path = document_name or ""
    file_name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    # schema contract: entities are grouped by layer buckets
    buckets: Dict[str, List[Dict[str, Any]]] = {b[1]: [] for b in _LAYER_BUCKETS}
    for ent in entities:
        buckets.setdefault(ent["layer"], []).append(ent)
    return {
        "document": {
            "file_name": file_name,
            "file_path": file_path,
            "domain": domain or "ENTERPRISE",
        },
        "layer_scope": "spec",
        "entities": buckets,
        "relations": relations,
        "evidence": [],
        "provenance": {"extractor_version": "packet-adapter/1.0"},
        "quality": {},
        "warnings": warnings,
        "assumptions": [],
    }


def iter_packet_entities(packet: Dict[str, Any]):
    """Yield every entity across layer buckets of a packet (read-order)."""
    for bucket in (packet.get("entities") or {}).values():
        if isinstance(bucket, list):
            yield from (e for e in bucket if isinstance(e, dict))


def maybe_attach_ontology_packet(result: Any) -> None:
    """Attach an OntologyPacket view to an extraction result when opted in.

    Dual-track protocol emission (ENTITY_CLASSIFICATION_ROADMAP T2.1): when
    ``EXTRACT_OUTPUT_PROTOCOL=packet``, rebuild the legacy layer-bucket view
    from *result* (duck-typed: ``l0_entities``/.../``relationships`` +
    optional ``metadata.source_path``/``metadata.domain``), adapt it, and
    expose it as ``result.ontology_packet``. Purely additive — legacy
    attributes stay untouched. Any failure is swallowed with a warning:
    protocol emission must never break extraction.
    """
    if os.environ.get("EXTRACT_OUTPUT_PROTOCOL", "").strip().lower() != "packet":
        return
    try:
        metadata = getattr(result, "metadata", {}) or {}
        legacy_view = {
            "L0_entities": list(getattr(result, "l0_entities", None) or []),
            "L1_entities": list(getattr(result, "l1_entities", None) or []),
            "L2_entities": list(getattr(result, "l2_entities", None) or []),
            "L2_5_entities": list(getattr(result, "l25_entities", None) or []),
            "L3_entities": list(getattr(result, "l3_entities", None) or []),
            "relationships": list(getattr(result, "relationships", None) or []),
        }
        packet = adapt_extraction_json_to_packet(
            legacy_view,
            document_name=metadata.get("source_path", ""),
            domain=metadata.get("domain", "") or "",
        )
        setattr(result, "ontology_packet", packet)
        n_entities = sum(len(v) for v in (packet.get("entities") or {}).values())
        logger.info(
            "[EXTRACT][PACKET] attached ontology_packet: "
            f"entities={n_entities} "
            f"relations={len(packet.get('relations', []))} "
            f"warnings={len(packet.get('warnings', []))}"
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"[EXTRACT][PACKET] attach failed (ignored): {exc}")
