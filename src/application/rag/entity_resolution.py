"""Entity canonicalization with reversible merges (ROADMAP T4.2).

Blocking → Matching → Clustering over packet entities, plus a Palantir-style
reversibility guarantee: ``apply_canonicalization`` never destroys anything —
the returned record carries a full snapshot of the pre-merge entity buckets
and relations, so ``undo_canonicalization`` restores the exact original
packet (deep-equal, not just semantically equal).

Matching rules are deterministic and explainable: two entities are merge
candidates iff they share layer+type AND their normalized names match or
their alias sets intersect. The winner of a cluster is the member with the
highest confidence (ties broken by lexicographically smallest id) — same
input always yields the same output.

Pure functions, stdlib only, cycle-safe.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

__all__ = [
    "normalize_entity_name",
    "block_entities",
    "cluster_entities",
    "apply_canonicalization",
    "undo_canonicalization",
]

_MAX_DEPTH = 64


def normalize_entity_name(name: Any) -> str:
    """Lowercase; drop spaces/underscores/dashes — comparison key only."""
    return str(name or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _alias_set(ent: Dict[str, Any]) -> set:
    out = {normalize_entity_name(ent.get("name"))}
    for a in ent.get("aliases") or []:
        n = normalize_entity_name(a)
        if n:
            out.add(n)
    out.discard("")
    return out


def _blocking_keys(ent: Dict[str, Any]) -> List[Tuple[str, str]]:
    """One block key per name/alias first-char — so alias-only duplicates
    whose names start differently still land in a common block."""
    etype = str(ent.get("type") or "")
    return [(etype, norm[:1] if norm else "#") for norm in sorted(_alias_set(ent))]


def block_entities(entities: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[int]]:
    """Group entity *indices* by multi-key blocks (name + alias first-chars).

    Blocking bounds pairwise matching cost; an entity may belong to several
    blocks (one per alias), guaranteeing any pair sharing layer+type and
    either name-normalization or alias overlap shares at least one block.
    """
    blocks: Dict[Tuple[str, str], List[int]] = {}
    for i, ent in enumerate(entities):
        for key in _blocking_keys(ent):
            blocks.setdefault(key, []).append(i)
    return blocks


def _match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if (a.get("layer"), a.get("type")) != (b.get("layer"), b.get("type")):
        return False
    na = normalize_entity_name(a.get("name"))
    nb = normalize_entity_name(b.get("name"))
    if na and na == nb:
        return True
    return bool(_alias_set(a) & _alias_set(b))


def cluster_entities(entities: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Union-find clusters over matched pairs; singletons included."""
    n = len(entities)
    parent: List[int] = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for members_idx in block_entities(entities).values():
        for x in range(len(members_idx)):
            for y in range(x + 1, len(members_idx)):
                ia, ib = members_idx[x], members_idx[y]
                if _match(entities[ia], entities[ib]):
                    union(ia, ib)

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)
    return [[entities[i] for i in sorted(members)] for members in groups.values()]


def _flatten_entities(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entity list from a packet; tolerates bucketed and flat shapes."""
    ents_field = packet.get("entities")
    if isinstance(ents_field, dict):
        out: List[Dict[str, Any]] = []
        for bucket in ents_field.values():
            if isinstance(bucket, list):
                out.extend(e for e in bucket if isinstance(e, dict))
        return out
    if isinstance(ents_field, list):
        return [e for e in ents_field if isinstance(e, dict)]
    return []


def _winner(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    def sort_key(e: Dict[str, Any]):
        conf = e.get("confidence")
        conf_v = conf if isinstance(conf, (int, float)) else -1.0
        # highest confidence first; tie-break smallest id for determinism
        return (-conf_v, str(e.get("id") or ""))

    return sorted(members, key=sort_key)[0]


def apply_canonicalization(packet: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Merge duplicate entities; return (new_packet, merge_record). Pure.

    The record embeds a deep-copy snapshot of the original buckets+relations,
    the cluster decisions, and every endpoint rewrite — sufficient for
    ``undo_canonicalization`` to restore the input bit-exactly.
    """
    ents_field = packet.get("entities") or {}
    flat = _flatten_entities(packet)

    snapshot = {
        "entities": copy.deepcopy(ents_field),
        "relations": copy.deepcopy(packet.get("relations")),
    }

    new_flat: List[Dict[str, Any]] = []
    id_map: Dict[str, str] = {}   # merged-away id -> canonical id
    decisions: List[Dict[str, Any]] = []

    for cluster in cluster_entities(flat):
        if len(cluster) == 1:
            keep = cluster[0]
            new_flat.append(copy.deepcopy(keep))
            continue
        win = _winner(cluster)
        win_id = str(win.get("id") or "")
        member_ids = [str(m.get("id") or "") for m in cluster]
        for mid in member_ids:
            id_map[mid] = win_id
        decisions.append({
            "canonical_id": win_id,
            "member_ids": member_ids,
            "members": copy.deepcopy(cluster),
            "matched_by": [
                f"layer={win.get('layer')} type={win.get('type')} "
                f"name_norm={normalize_entity_name(win.get('name'))}"
            ],
        })
        new_flat.append(copy.deepcopy(win))

    relations = copy.deepcopy(packet.get("relations")) or []
    rewrites: List[Dict[str, Any]] = []
    for idx, rel in enumerate(relations):
        for field in ("source_id", "target_id"):
            old = rel.get(field)
            if isinstance(old, str) and old in id_map and id_map[old] != old:
                new_val = id_map[old]
                rewrites.append({
                    "relation_index": idx, "field": field, "from": old, "to": new_val,
                })
                rel[field] = new_val
        # dedupe relations that collapsed onto themselves after rewrite
    deduped: List[Dict[str, Any]] = []
    seen: set = set()
    for rel in relations:
        key = (
            rel.get("source_id"), rel.get("relation_type"), rel.get("target_id"),
            rel.get("evidence_ref"),
        )
        if key in seen:
            rewrites.append({"collapsed_duplicate": list(key)})
            continue
        seen.add(key)
        deduped.append(rel)

    new_packet = copy.deepcopy(packet)
    if isinstance(ents_field, dict):
        buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ents_field}
        for e in new_flat:
            buckets.setdefault(str(e.get("layer") or ""), []).append(e)
        new_packet["entities"] = buckets
    else:
        new_packet["entities"] = new_flat
    new_packet["relations"] = deduped

    record = {
        "snapshot": snapshot,
        "decisions": decisions,
        "endpoint_rewrites": rewrites,
        "id_map": id_map,
    }
    return new_packet, record


def undo_canonicalization(packet: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """Restore the exact pre-merge packet from a merge record. Pure."""
    restored = copy.deepcopy(packet)
    restored["entities"] = copy.deepcopy(record["snapshot"]["entities"])
    restored["relations"] = copy.deepcopy(record["snapshot"]["relations"])
    return restored
