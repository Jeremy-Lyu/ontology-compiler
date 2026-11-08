"""OntologyPacket → StructuredIndex bridge (RAG roadmap T4.1).

The extraction packet remains the canonical interchange contract. This module
adapts packet entities and relations to the existing Pydantic-backed
``StructuredIndex`` without changing that index's public API. The bridge is
explicit and deterministic: callers decide whether/how to persist the index.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.domain.models.canonical import CanonicalEntity, CanonicalRelation

from .structured_index import StructuredIndex
from .packet_derivation import derive_evidence_bundles

__all__ = ["canonical_models_from_packet", "index_packet"]


def _entities(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    field = packet.get("entities") or {}
    if isinstance(field, dict):
        out: List[Dict[str, Any]] = []
        for bucket in field.values():
            if isinstance(bucket, list):
                out.extend(x for x in bucket if isinstance(x, dict))
        return out
    return [x for x in field if isinstance(x, dict)] if isinstance(field, list) else []


def canonical_models_from_packet(
    packet: Dict[str, Any],
) -> Dict[str, List[Any]]:
    """Convert packet records to existing Canonical model instances.

    Packet ``attributes`` are canonical dicts; CanonicalEntity accepts them as
    properties while preserving the original shape in ``metadata``. Relation
    evidence refs are retained in relation metadata because the legacy
    CanonicalRelation evidence model requires a richer source object.
    """
    entities: List[CanonicalEntity] = []
    for raw in _entities(packet):
        payload = dict(raw)
        payload["entity_id"] = payload.get("id", "")
        payload["entity_type"] = payload.get("type", "")
        payload["properties"] = payload.get("attributes") if isinstance(payload.get("attributes"), dict) else {}
        payload["metadata"] = {
            "layer_scope": payload.get("layer_scope", "spec"),
            "evidence_ref": payload.get("evidence_ref", ""),
            "source_file": payload.get("source_file", ""),
        }
        entities.append(CanonicalEntity.from_dict(payload))

    relations: List[CanonicalRelation] = []
    for idx, raw in enumerate(packet.get("relations") or []):
        if not isinstance(raw, dict):
            continue
        payload = dict(raw)
        payload["relation_id"] = payload.get(
            "relation_id",
            f"rel:{payload.get('source_id', '')}:{payload.get('relation_type', '')}:{payload.get('target_id', '')}:{idx}",
        )
        payload["properties"] = dict(payload.get("attributes") or {})
        payload["metadata"] = {
            "layer_scope": payload.get("layer_scope", "spec"),
            "evidence_ref": payload.get("evidence_ref", ""),
            "source_cq": payload.get("source_cq", ""),
            "effective_from": payload.get("effective_from", ""),
            "effective_to": payload.get("effective_to", ""),
        }
        relations.append(CanonicalRelation.from_dict(payload))
    return {"entities": entities, "relations": relations}


def index_packet(packet: Dict[str, Any], index: StructuredIndex | None = None) -> StructuredIndex:
    """Index packet entities/relations and evidence bundles in ``StructuredIndex``.

    ``index`` may be supplied for incremental use; otherwise a new index is
    returned. Entity and relation indexing always occurs. Evidence bundles are
    retained in ``index`` only when the project's CanonicalEvidence contract
    can represent a source ref; the packet evidence bundle itself remains
    available through ``derive_evidence_bundles`` for citation consumers.
    """
    target = index or StructuredIndex()
    models = canonical_models_from_packet(packet)
    names = {e.entity_id: e.name for e in models["entities"]}
    for entity in models["entities"]:
        target.index_entity(entity)
    for relation in models["relations"]:
        target.index_relation(
            relation,
            from_name=names.get(relation.source_id, ""),
            to_name=names.get(relation.target_id, ""),
        )
    return target
