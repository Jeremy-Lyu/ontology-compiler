"""Global cross-file meta_id collision fix extracted from extract stage.

Phase 3 LOC reduction (docs/CODE_CLEANUP_CANDIDATES.md §3.1): the
[GLOBAL ID FIX] cluster moved verbatim from src/pipeline/stages/extract.py.
Self-contained by design: pure functions over node/relationship lists,
stdlib only. extract.py re-imports these names so internal call sites and
any `extract._apply_global_id_uniqueness` style access keep working.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def build_global_id_indexes(all_nodes: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Build name->id and id->node indexes over all nodes."""
    name_to_id = {}
    id_to_node = {}
    for n in all_nodes:
        iid = n.get("meta_id", "")
        nname = n.get("name", "").strip()
        id_to_node[iid] = n
        if nname:
            name_to_id[nname.lower().strip()] = iid
    return name_to_id, id_to_node


def remap_colliding_node_ids(
    all_nodes: list[dict[str, Any]],
    collisions: set[str],
    name_to_id: dict[str, str],
    id_to_node: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Reassign colliding meta_ids to fresh layer-sequenced ids; returns old->new mapping."""
    layer_counters = {}
    id_mapping = {}
    for n in all_nodes:
        iid = n.get("meta_id", "")
        if iid in collisions:
            layer = n.get("layer", "L2")
            if layer not in layer_counters:
                existing = [x for x in all_nodes if x.get("layer") == layer]
                max_num = 0
                for x in existing:
                    try:
                        num = int(x.get("meta_id", "X-0").split("-")[-1])
                        max_num = max(max_num, num)
                    except (ValueError, IndexError):
                        pass
                layer_counters[layer] = max_num + 1
            new_id = f"{layer}-{layer_counters[layer]:03d}"
            layer_counters[layer] += 1
            old_id = iid
            n["meta_id"] = new_id
            id_mapping[old_id] = new_id
            id_to_node[new_id] = n
            name_to_id[n.get("name", "").lower().strip()] = new_id
            print(f"[Extract] [GLOBAL ID FIX] {old_id} -> {new_id} for '{n.get('name','')}'")
    return id_mapping


def rewire_colliding_relationships(
    all_relationships: list[dict[str, Any]],
    id_mapping: dict[str, str],
    name_to_id: dict[str, str],
    id_to_node: dict[str, dict[str, Any]],
) -> None:
    """Update relationship endpoints to point at the reassigned ids."""
    for rel in all_relationships:
        from_id = rel.get("from", rel.get("from_id", ""))
        to_id = rel.get("to", rel.get("to_id", ""))
        from_name = rel.get("from_name", "")
        to_name = rel.get("to_name", "")

        if from_id in id_mapping:
            if from_name:
                name_key = from_name.lower().strip()
                if name_key in name_to_id:
                    rel["from"] = name_to_id[name_key]
                else:
                    rel["from"] = id_mapping[from_id]
            else:
                rel["from"] = id_mapping[from_id]
        elif from_id in id_to_node and from_name:
            actual_name = id_to_node[from_id].get("name", "").strip()
            if actual_name and actual_name.lower() != from_name.lower():
                name_key = from_name.lower().strip()
                if name_key in name_to_id:
                    rel["from"] = name_to_id[name_key]

        if to_id in id_mapping:
            if to_name:
                name_key = to_name.lower().strip()
                if name_key in name_to_id:
                    rel["to"] = name_to_id[name_key]
                else:
                    rel["to"] = id_mapping[to_id]
            else:
                rel["to"] = id_mapping[to_id]
        elif to_id in id_to_node and to_name:
            actual_name = id_to_node[to_id].get("name", "").strip()
            if actual_name and actual_name.lower() != to_name.lower():
                name_key = to_name.lower().strip()
                if name_key in name_to_id:
                    rel["to"] = name_to_id[name_key]


def report_global_id_uniqueness(all_nodes: list[dict[str, Any]]) -> None:
    """Verify and report whether any meta_id collisions remain."""
    _new_counts = Counter(n.get("meta_id", "") for n in all_nodes if isinstance(n, dict))
    _remaining = {iid for iid, cnt in _new_counts.items() if cnt > 1 and iid}
    if _remaining:
        print(f"[Extract] [GLOBAL ID FIX] WARNING: Still have collisions: {_remaining}")
    else:
        print(f"[Extract] [GLOBAL ID FIX] All ID collisions resolved ✅")


def apply_global_id_uniqueness(all_nodes: list[dict[str, Any]], all_relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _id_counts = Counter(n.get("meta_id", "") for n in all_nodes if isinstance(n, dict))
    _collisions = {iid for iid, cnt in _id_counts.items() if cnt > 1 and iid}
    if not _collisions:
        return all_nodes

    print(f"[Extract] [GLOBAL ID FIX] Detected {len(_collisions)} cross-file ID collisions: {_collisions}")
    name_to_id, id_to_node = build_global_id_indexes(all_nodes)
    id_mapping = remap_colliding_node_ids(all_nodes, _collisions, name_to_id, id_to_node)
    rewire_colliding_relationships(all_relationships, id_mapping, name_to_id, id_to_node)
    report_global_id_uniqueness(all_nodes)
    return all_nodes


# Historical underscore-prefixed names kept as aliases so extract.py's
# internal call sites and any external access stay unchanged.
_build_global_id_indexes = build_global_id_indexes
_remap_colliding_node_ids = remap_colliding_node_ids
_rewire_colliding_relationships = rewire_colliding_relationships
_report_global_id_uniqueness = report_global_id_uniqueness
_apply_global_id_uniqueness = apply_global_id_uniqueness
