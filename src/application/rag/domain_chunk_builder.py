"""Domain-ontology chunk builder (bounded context for layer1_domain_ontology.json).

[P2 RAG 收口 2026-09-03]: extracted from chunk_indexing.py so the layer-1
domain ontology -> ChunkRecord pipeline owns its bounded context:
- capability level / number / parent inference (infer_capability_level, ...)
- capability / path / system-mapping card builders
- generic linked-entity card builder (data / actor / asset / scenario)
- build_chunks_from_domain_ontology entry point
  (re-exported by chunk_indexing / rag package for compatibility)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.domain.errors import DataContractError

from .domain.models import (
    CHUNK_PREFIX_ACTOR,
    CHUNK_PREFIX_ASSET,
    CHUNK_PREFIX_CAP,
    CHUNK_PREFIX_DATA,
    CHUNK_PREFIX_PATH,
    CHUNK_PREFIX_SCENARIO,
    CHUNK_PREFIX_SYSMAP,
    CHUNK_TYPE_ACTOR_CARD,
    CHUNK_TYPE_ASSET_CARD,
    CHUNK_TYPE_CAPABILITY_CARD,
    CHUNK_TYPE_DATA_CARD,
    CHUNK_TYPE_PATH_CARD,
    CHUNK_TYPE_SCENARIO_CARD,
    CHUNK_TYPE_SYSTEM_MAPPING_CARD,
    EDGE_TYPES_SYSTEM_TO_CAP as _EDGE_TYPES_SYSTEM_TO_CAP,
    ChunkRecord,
)


def _require_list(payload: Dict[str, Any], key: str, source: str) -> List[Dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise DataContractError(f"missing/invalid '{key}' list in {source}")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise DataContractError(f"invalid '{key}[{i}]' record in {source}")
        out.append(item)
    return out


def _node_id(node: Dict[str, Any], source: str) -> str:
    mid = node.get("meta_id")
    if not isinstance(mid, str) or not mid.strip():
        raise DataContractError(f"node missing meta_id in {source}")
    return mid


def _make_chunk_id(prefix: str, node_id: str) -> str:
    """Stable chunk-ID policy (RAG-03).

    Single source of truth for chunk identifiers: ``prefix + node_id``.
    All card builders (capability / path / system-mapping / linked data-actor-
    asset-scenario) must derive their chunk_id here so the rule stays unique.
    The per-type prefixes (CHUNK_PREFIX_*) guarantee that distinct card types
    never collide even when they reference the same node.
    """
    return f"{prefix}{node_id}"


# =============================================================================
# Capability Level Inference
# =============================================================================

_CAP_L1_RE = re.compile(r"^CAP_L1_(?P<l1>\d+)_")
_CAP_L2_RE = re.compile(r"^CAP_L2_(?P<l1>\d+)_(?P<l2>\d+)_")
_CAP_L3_RE = re.compile(r"^CAP_L3_(?P<l1>\d+)_(?P<l2>\d+)_(?P<l3>\d+)_")


def infer_capability_level(cap_id: str, node: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Infer capability level (L1/L2/L3) from node id or node layer field.

    Priority:
    1. Node layer field (highest priority, most reliable)
    2. ID pattern matching (CAP_L1_, CAP_L2_, CAP_L3_)
    3. Fallback: count underscore-separated numeric segments
    4. Final fallback: return None if no pattern matches

    Examples:
        CAP_L1_2_PRODUCT_DEVELOPMENT -> "L1"
        CAP_L2_2_4_PRODUCT_VERIFICATION -> "L2"
        CAP_L3_2_1_3_PRODUCT_PART_STRUCTURE -> "L3"
        CAP_2_PRODUCT_DEVELOPMENT -> "L1" (fallback)
        CAP_2_4_PRODUCT_VERIFICATION -> "L2" (fallback)
        SYS_X -> None
        node['layer']='L2' -> "L2" (highest priority)
    """
    if node and "layer" in node:
        layer = str(node["layer"]).strip().upper()
        if layer in ("L1", "L2", "L3", "L0", "L2.5"):
            return layer

    if _CAP_L1_RE.match(cap_id):
        return "L1"
    if _CAP_L2_RE.match(cap_id):
        return "L2"
    if _CAP_L3_RE.match(cap_id):
        return "L3"

    parts = cap_id.split("_")
    num_parts = [p for p in parts if p.isdigit()]
    if len(num_parts) == 1:
        return "L1"
    elif len(num_parts) == 2:
        return "L2"
    elif len(num_parts) >= 3:
        return "L3"

    return None


def infer_parent_capability_ids(cap_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer parent capability ids for a capability node.

    Returns:
        (l1_id, l2_id)

    Notes:
        This relies on the current ID convention in ontology artifacts.
    """

    m3 = _CAP_L3_RE.match(cap_id)
    if m3:
        l1 = m3.group("l1")
        l2 = m3.group("l2")
        l1_id = f"CAP_L1_{l1}_"  # prefix only (resolved later)
        l2_id = f"CAP_L2_{l1}_{l2}_"  # prefix only
        return l1_id, l2_id

    m2 = _CAP_L2_RE.match(cap_id)
    if m2:
        l1 = m2.group("l1")
        l1_id = f"CAP_L1_{l1}_"
        return l1_id, None

    m1 = _CAP_L1_RE.match(cap_id)
    if m1:
        return None, None

    return None, None


def infer_capability_number(cap_id: str) -> Optional[str]:
    """Infer human-facing capability number (e.g., '2', '2.1', '2.1.3') from node id."""

    m1 = _CAP_L1_RE.match(cap_id)
    if m1:
        return str(int(m1.group("l1")))

    m2 = _CAP_L2_RE.match(cap_id)
    if m2:
        return f"{int(m2.group('l1'))}.{int(m2.group('l2'))}"

    m3 = _CAP_L3_RE.match(cap_id)
    if m3:
        return f"{int(m3.group('l1'))}.{int(m3.group('l2'))}.{int(m3.group('l3'))}"

    return None


def _resolve_prefix_to_node_id(prefix: str, all_node_ids: Sequence[str]) -> Optional[str]:
    matches = [nid for nid in all_node_ids if nid.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return None


# =============================================================================
# Chunk Building Functions
# =============================================================================

def _build_capability_chunks(node_by_id: Dict[str, Dict[str, Any]], all_node_ids: Sequence[str], source_file: str) -> List[ChunkRecord]:
    """Build capability cards + path cards for capability nodes."""
    chunks: List[ChunkRecord] = []
    for nid, n in node_by_id.items():
        if n.get("meta_class") != "Capability":
            continue

        level = infer_capability_level(nid, n)
        cap_num = infer_capability_number(nid)
        meta_name = str(n.get("meta_name") or "")
        meta_desc = str(n.get("meta_description") or "")
        owner = str(n.get("meta_owner") or "")
        domain = str(n.get("sandbox_domain") or "")

        linked: List[str] = []
        l1_prefix, l2_prefix = infer_parent_capability_ids(nid)
        if l1_prefix:
            resolved_l1 = _resolve_prefix_to_node_id(l1_prefix, all_node_ids)
            if resolved_l1:
                linked.append(resolved_l1)
        if l2_prefix:
            resolved_l2 = _resolve_prefix_to_node_id(l2_prefix, all_node_ids)
            if resolved_l2:
                linked.append(resolved_l2)

        content = "\n".join(
            [
                f"Capability: {meta_name}",
                f"ID: {nid}",
                f"Number: {cap_num or ''}",
                f"Level: {level or ''}",
                f"Domain: {domain}",
                f"Owner: {owner}",
                f"Description: {meta_desc}",
            ]
        ).strip()

        chunks.append(
            ChunkRecord(
                chunk_id=_make_chunk_id(CHUNK_PREFIX_CAP, nid),
                chunk_type=CHUNK_TYPE_CAPABILITY_CARD,
                content=content,
                node_ids=[nid],
                linked_node_ids=linked,
                meta={
                    "level": level,
                    "domain": domain,
                    "owner": owner,
                    "source_file": source_file,
                    "meta_class": "Capability",
                    "meta_subclass": n.get("meta_subclass"),
                },
            )
        )

        if linked:
            path = " -> ".join([
                *(node_by_id[lid].get("meta_name") for lid in linked if lid in node_by_id),
                meta_name,
            ]).strip()
            chunks.append(
                ChunkRecord(
                    chunk_id=_make_chunk_id(CHUNK_PREFIX_PATH, nid),
                    chunk_type=CHUNK_TYPE_PATH_CARD,
                    content=path,
                    node_ids=[nid],
                    linked_node_ids=linked,
                    meta={
                        "level": level,
                        "domain": domain,
                        "source_file": source_file,
                        "chunk_type": CHUNK_TYPE_PATH_CARD,
                    },
                )
            )
    return chunks


def _build_system_mapping_chunks(sys_to_caps: Dict[str, List[Dict[str, Any]]], node_by_id: Dict[str, Dict[str, Any]], source_file: str) -> List[ChunkRecord]:
    """Build system mapping cards from the system->capability adjacency."""
    chunks: List[ChunkRecord] = []
    for sys_id, rel_edges in sys_to_caps.items():
        sys_node = node_by_id.get(sys_id)
        if not sys_node:
            continue
        sys_name = str(sys_node.get("meta_name") or "")
        sys_desc = str(sys_node.get("meta_description") or "")
        sys_domain = str(sys_node.get("sandbox_domain") or "")

        targets: List[str] = []
        edge_types: List[str] = []
        for e in rel_edges:
            tgt = e.get("target")
            if isinstance(tgt, str):
                targets.append(tgt)
            et = e.get("type")
            if isinstance(et, str):
                edge_types.append(et)

        content_lines = [
            f"System: {sys_name}",
            f"Domain: {sys_domain}",
            f"Description: {sys_desc}",
            f"Mappings ({len(targets)}):",
        ]
        for t in sorted(set(targets))[:50]:
            tnode = node_by_id.get(t)
            tname = str(tnode.get("meta_name") or "") if tnode else ""
            content_lines.append(f"- {t} {tname}")

        chunks.append(
            ChunkRecord(
                chunk_id=_make_chunk_id(CHUNK_PREFIX_SYSMAP, sys_id),
                chunk_type=CHUNK_TYPE_SYSTEM_MAPPING_CARD,
                content="\n".join(content_lines).strip(),
                node_ids=[sys_id],
                linked_node_ids=sorted(set(targets)),
                meta={
                    "domain": sys_domain,
                    "owner": sys_node.get("meta_owner"),
                    "source_file": source_file,
                    "edge_types": sorted(set(edge_types)),
                    "meta_class": "System",
                    "meta_subclass": sys_node.get("meta_subclass"),
                },
            )
        )
    return chunks


def _collect_domain_chunk_lists(node_by_id: Dict[str, Dict[str, Any]], relationships: Sequence[Dict[str, Any]], source_file: str) -> List[ChunkRecord]:
    """Build capability/path/system chunks (the upper half of the domain ontology)."""
    all_node_ids = list(node_by_id.keys())

    # system -> capability adjacency for mapping cards
    sys_to_caps: Dict[str, List[Dict[str, Any]]] = {}
    for rel in relationships:
        src = rel.get("source")
        tgt = rel.get("target")
        etype = rel.get("type")
        if not isinstance(src, str) or not isinstance(tgt, str) or not isinstance(etype, str):
            continue
        if etype.upper() not in _EDGE_TYPES_SYSTEM_TO_CAP:
            continue
        src_node = node_by_id.get(src)
        tgt_node = node_by_id.get(tgt)
        if not src_node or not tgt_node:
            continue
        if src_node.get("meta_class") != "System":
            continue
        if tgt_node.get("meta_class") != "Capability":
            continue
        sys_to_caps.setdefault(src, []).append(rel)

    chunks: List[ChunkRecord] = _build_capability_chunks(node_by_id, all_node_ids, source_file)
    chunks.extend(_build_system_mapping_chunks(sys_to_caps, node_by_id, source_file))
    return chunks


# Card configs for the generic linked-entity builder.
# (meta_classes, card_prefix, card_type, content_label, scenario)
_LINKED_CARD_CONFIGS = [
    ({"Data"}, CHUNK_PREFIX_DATA, CHUNK_TYPE_DATA_CARD, "Data", False),
    ({"Actor"}, CHUNK_PREFIX_ACTOR, CHUNK_TYPE_ACTOR_CARD, "Actor", False),
    ({"Asset"}, CHUNK_PREFIX_ASSET, CHUNK_TYPE_ASSET_CARD, "Asset", False),
    ({"BusinessScenario", "Scenario", "UseCase"}, CHUNK_PREFIX_SCENARIO, CHUNK_TYPE_SCENARIO_CARD, "Scenario", True),
]


def _build_linked_card_chunks(
    node_by_id: Dict[str, Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
    source_file: str,
    *,
    meta_classes: set,
    card_prefix: str,
    card_type: str,
    content_label: str,
    scenario: bool = False,
) -> List[ChunkRecord]:
    """Build a linked-entity card per matching node (data/actor/asset/scenario)."""
    chunks: List[ChunkRecord] = []
    for nid, n in node_by_id.items():
        mc = n.get("meta_class")
        if mc not in meta_classes:
            continue

        name = str(n.get("meta_name") or "")
        desc = str(n.get("meta_description") or "")
        domain = str(n.get("sandbox_domain") or "")
        owner = str(n.get("meta_owner") or "")
        layer = str(n.get("layer") or "L2") if scenario else ""

        linked: List[str] = []
        rel_types: List[str] = []
        for rel in relationships:
            src = rel.get("source")
            tgt = rel.get("target")
            rtype = rel.get("type")
            if not isinstance(src, str) or not isinstance(tgt, str) or not isinstance(rtype, str):
                continue
            if src == nid:
                linked.append(tgt)
                rel_types.append(rtype)
            elif tgt == nid:
                linked.append(src)
                rel_types.append(rtype)

        content_lines = [f"{content_label}: {name}"]
        if scenario:
            content_lines.append(f"Type: {mc}")
            content_lines.append(f"Layer: {layer}")
        content_lines.extend([f"Domain: {domain}", f"Owner: {owner}", f"Description: {desc}"])
        if linked:
            content_lines.append(f"Related entities ({len(set(linked))}):")
            for lid in sorted(set(linked))[:20]:
                lnode = node_by_id.get(lid)
                lname = str(lnode.get("meta_name") or "") if lnode else ""
                content_lines.append(f"- {lid} {lname}")

        meta: Dict[str, Any] = {
            "domain": domain,
            "owner": owner,
            "source_file": source_file,
            "relationship_types": sorted(set(rel_types)),
            "meta_class": mc,
            "meta_subclass": n.get("meta_subclass"),
        }
        if scenario:
            meta["layer"] = layer

        chunks.append(
            ChunkRecord(
                chunk_id=_make_chunk_id(card_prefix, nid),
                chunk_type=card_type,
                content="\n".join(content_lines).strip(),
                node_ids=[nid],
                linked_node_ids=sorted(set(linked)),
                meta=meta,
            )
        )
    return chunks


def _collect_domain_other_chunks(node_by_id: Dict[str, Dict[str, Any]], relationships: Sequence[Dict[str, Any]], source_file: str) -> List[ChunkRecord]:
    """Build data/actor/asset/scenario chunks via the generic linked-card builder."""
    chunks: List[ChunkRecord] = []
    for meta_classes, prefix, card_type, label, scenario in _LINKED_CARD_CONFIGS:
        chunks.extend(
            _build_linked_card_chunks(
                node_by_id, relationships, source_file,
                meta_classes=meta_classes, card_prefix=prefix, card_type=card_type,
                content_label=label, scenario=scenario,
            )
        )
    return chunks

def build_chunks_from_domain_ontology(payload: Dict[str, Any], source_file: str) -> List[ChunkRecord]:
    """Build chunks from layer1_domain_ontology.json payload."""

    nodes = _require_list(payload, "nodes", source_file)
    # R18 Fragment 模型标准化: 使用 'relationships' 作为标准字段名
    relationships = _require_list(payload, "relationships", source_file)

    node_by_id: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        node_by_id[_node_id(n, source_file)] = n

    chunks = _collect_domain_chunk_lists(node_by_id, relationships, source_file)
    chunks.extend(_collect_domain_other_chunks(node_by_id, relationships, source_file))
    return chunks
