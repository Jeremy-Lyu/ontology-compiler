"""RAG consumption-object derivation from OntologyPacket (ROADMAP T4.1).

Phase 4 of the entity-classification roadmap: the packet is the *storage*
contract; RAG should not index raw extraction JSON but derived, readable,
explainable retrieval objects:

- Canonical Entity   → graph-store shape (stable identity + attributes)
- RAG Chunk Entity   → retrievable unit (title/summary/keywords/parent path)
- Evidence Bundle    → citation unit (span refs grouped per document)

All functions are pure, dependency-light (stdlib only) and cycle-safe:
parent_id chains are walked with a visited set, so malformed or looping
hierarchies degrade to a truncated path instead of hanging.
"""
from __future__ import annotations

from typing import Any, Dict, List

__all__ = [
    "derive_canonical_entity",
    "derive_rag_chunk_entity",
    "derive_evidence_bundles",
    "derive_rag_objects",
]

_MAX_PARENT_DEPTH = 16


def derive_canonical_entity(ent: Dict[str, Any]) -> Dict[str, Any]:
    """Graph-store projection of one packet entity."""
    return {
        "id": ent.get("id") or "",
        "type": ent.get("type") or "",
        "sub_type": ent.get("sub_type") or "",
        "name": ent.get("name") or "",
        "layer": ent.get("layer") or "",
        "status": ent.get("status") or "",
        "version": ent.get("version") or "",
        "attributes": ent.get("attributes") if isinstance(ent.get("attributes"), dict) else {},
        "aliases": list(ent.get("aliases") or []),
        "domain": ent.get("domain") or "",
    }


def _parent_path(ent: Dict[str, Any], entities_by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    """Walk parent_id chain root→self; cycle-safe via visited set."""
    chain: List[str] = [ent.get("name") or ent.get("id") or ""]
    seen = {ent.get("id")}
    cur = entities_by_id.get(ent.get("parent_id") or "")
    depth = 0
    while cur is not None and depth < _MAX_PARENT_DEPTH:
        label = cur.get("name") or cur.get("id") or ""
        chain.append(label)
        cid = cur.get("id")
        if cid in seen:
            break
        seen.add(cid)
        cur = entities_by_id.get(cur.get("parent_id") or "")
        depth += 1
    chain.reverse()  # root first
    return [seg for seg in chain if seg]


def derive_rag_chunk_entity(
    ent: Dict[str, Any],
    entities_by_id: Dict[str, Dict[str, Any]],
    *,
    document_name: str = "",
) -> Dict[str, Any]:
    """Retriable unit projected from one packet entity."""
    keywords: List[str] = []
    for kw in list(ent.get("aliases") or []) + list(ent.get("tags") or []):
        kw = str(kw).strip()
        if kw and kw not in keywords:
            keywords.append(kw)
    return {
        "chunk_id": f"ragc:{ent.get('id') or ''}",
        "entity_id": ent.get("id") or "",
        "title": ent.get("name") or "",
        "summary": ent.get("description") or "",
        "keywords": keywords,
        "layer": ent.get("layer") or "",
        "type": ent.get("type") or "",
        "parent_path": _parent_path(ent, entities_by_id),
        "evidence_ref": ent.get("evidence_ref") or "",
        "source_file": document_name,
        "status": ent.get("status") or "",
        "valid_from": ent.get("valid_from") or "",
        "valid_to": ent.get("valid_to") or "",
    }


def derive_evidence_bundles(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Citation units grouped by evidence_ref across relations and entities.

    Entities/relations sharing the same evidence_ref land in one bundle so
    answers can cite once for multiple claims. Bundles with no usable ref
    are skipped.
    """
    doc = ((packet.get("document") or {}).get("file_name")) or ""
    groups: Dict[str, Dict[str, Any]] = {}

    def _touch(ref: str, kind: str, oid: str) -> None:
        ref = str(ref or "").strip()
        if not ref:
            return
        bundle = groups.setdefault(ref, {
            "id": f"evb:{len(groups) + 1}",
            "evidence_ref": ref,
            "document": doc,
            "entity_ids": [],
            "relation_ids": [],
        })
        key = "entity_ids" if kind == "entity" else "relation_ids"
        if oid and oid not in bundle[key]:
            bundle[key].append(oid)

    entities_field = packet.get("entities") or {}
    if isinstance(entities_field, dict):
        entity_buckets: Any = list(entities_field.values())
    elif isinstance(entities_field, list):
        entity_buckets = [entities_field]  # tolerate flat packets defensively
    else:
        entity_buckets = []
    for bucket in entity_buckets:
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if isinstance(item, dict):
                _touch(item.get("evidence_ref"), "entity", item.get("id"))

    for rel in packet.get("relations") or []:
        if isinstance(rel, dict):
            rid = f"{rel.get('source_id')}->{rel.get('relation_type')}->{rel.get('target_id')}"
            _touch(rel.get("evidence_ref"), "relation", rid)

    return sorted(groups.values(), key=lambda b: b["id"])


def derive_rag_objects(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Derive all three consumption objects from one packet. Pure function."""
    entities_field = packet.get("entities") or {}
    if isinstance(entities_field, dict):
        ents: List[Dict[str, Any]] = [
            e for bucket in entities_field.values()
            if isinstance(bucket, list)
            for e in bucket
            if isinstance(e, dict)
        ]
    elif isinstance(entities_field, list):
        ents = [e for e in entities_field if isinstance(e, dict)]
    else:
        ents = []
    by_id = {e.get("id"): e for e in ents}
    document_name = ((packet.get("document") or {}).get("file_name")) or ""

    canonical = [derive_canonical_entity(e) for e in ents]
    chunks = [derive_rag_chunk_entity(e, by_id, document_name=document_name) for e in ents]
    bundles = derive_evidence_bundles(packet)
    return {
        "canonical_entities": canonical,
        "rag_chunks": chunks,
        "evidence_bundles": bundles,
    }
