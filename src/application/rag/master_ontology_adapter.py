"""Legacy master_ontology directory → OntologyPacket adapter.

The current master export is a directory of legacy files:
``nodes.json`` (flat node list), ``relationships.json`` (source/target
relations), and ``instances.json`` (a separate L3 projection). This adapter
makes ``nodes.json`` + ``relationships.json`` usable by the packet-first RAG
bridges without duplicating RAG indexing logic.

Design rule: ``nodes.json`` is the canonical node source. ``instances.json``
is diagnostic/legacy output and is not appended, because doing so would
silently duplicate entity IDs already present in nodes.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

__all__ = ["load_master_ontology_packet"]

_LAYERS = ("L0", "L1", "L2", "L2.5", "L3")

# Retrieval-only bilingual alias table (canonical entity name -> Chinese/variant
# aliases). Kept separate from config/entity_aliases.json, which feeds the
# extract/entity-resolution chain; this one only enriches rag chunk keywords so
# Chinese business terms (质量/库存/价值流图...) can match English-named entities.
_BILINGUAL_ALIASES_PATH = Path(__file__).resolve().parents[3] / "config" / "entity_bilingual_aliases.json"


def _load_bilingual_aliases(path: Path | None = None) -> List[Dict[str, Any]]:
    """Load retrieval-only bilingual alias groups.

    Each group: {"canonical": str, "match": [name patterns], "aliases": [...]}.
    ``match`` defaults to [canonical]; a group applies to a node when the node
    name contains any match pattern (case-insensitive), so prefixed/suffixed
    master names ("Safety Stock Parameter", "VSM Map") inherit the vocabulary.
    """
    p = path or _BILINGUAL_ALIASES_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: List[Dict[str, Any]] = []
    items = data.get("aliases", []) if isinstance(data, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or "").strip()
        aliases = [str(a).strip() for a in item.get("aliases", []) if str(a).strip()]
        if not canonical or not aliases:
            continue
        match = [str(m).strip().lower() for m in item.get("match", []) or [] if str(m).strip()]
        out.append({"match": match or [canonical.lower()], "aliases": aliases})
    return out


def _read(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, type(default)) else default
    except (OSError, ValueError):
        return default


def _evidence_ref(node: Dict[str, Any]) -> str:
    evidence = node.get("evidence")
    if isinstance(evidence, dict):
        return str(evidence.get("content") or evidence.get("source_text") or "").strip()
    return str(node.get("evidence_ref") or "").strip()


def _attributes(node: Dict[str, Any]) -> Dict[str, Any]:
    raw = node.get("attributes")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        out: Dict[str, Any] = {}
        for item in raw:
            if isinstance(item, dict) and item.get("name"):
                out[str(item["name"])] = item.get("value")
        return out
    return {}


def _adapt_node(node: Dict[str, Any]) -> Dict[str, Any]:
    layer = str(node.get("layer") or "L1")
    if layer not in _LAYERS:
        layer = "L1"
    ent: Dict[str, Any] = {
        "id": str(node.get("meta_id") or node.get("entity_id") or "").strip(),
        "code": str(node.get("code") or "").strip(),
        "name": str(node.get("name") or node.get("meta_name") or "").strip(),
        "layer": layer,
        "layer_scope": "spec",
        "type": str(node.get("meta_class") or node.get("entity_type") or "").strip(),
        "sub_type": str(node.get("meta_subclass") or "").strip(),
        "description": str(node.get("description") or "").strip(),
        "domain": str(node.get("domain") or "ENTERPRISE").strip(),
        "source_file": str(node.get("source_file") or "").strip(),
        "source_span": str(node.get("position") or node.get("chunk_section") or "").strip(),
        "evidence_ref": _evidence_ref(node),
        "version": str(node.get("version") or "1.0"),
        "attributes": _attributes(node),
        "aliases": list(node.get("aliases") or []) if isinstance(node.get("aliases"), list) else [],
        "tags": [],
    }
    status = str(node.get("status") or "").strip()
    if status in {"active", "deprecated", "proposed", "draft", "ghost", "archived"}:
        ent["status"] = status
    confidence = node.get("confidence")
    if isinstance(confidence, (int, float)):
        ent["confidence"] = confidence
    return ent


def _adapt_relation(rel: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "source_id": str(rel.get("source") or rel.get("source_id") or "").strip(),
        "target_id": str(rel.get("target") or rel.get("target_id") or "").strip(),
        "relation_type": str(rel.get("relation_type") or rel.get("type") or "").strip(),
        "evidence_ref": str(rel.get("evidence_ref") or "").strip(),
    }
    confidence = rel.get("confidence")
    if isinstance(confidence, (int, float)):
        out["confidence"] = confidence
    provenance = rel.get("provenance")
    if isinstance(provenance, dict):
        out["provenance"] = "rule" if provenance.get("stage") == "dedup" else "llm"
    elif provenance in {"llm", "skill", "rule", "inferred"}:
        out["provenance"] = provenance
    return out


def load_master_ontology_packet(master_dir: Path | str) -> Dict[str, Any]:
    """Load the current master ontology directory as a packet-first input."""
    root = Path(master_dir)
    raw_nodes = _read(root / "nodes.json", [])
    raw_relations = _read(root / "relationships.json", [])
    entities = [_adapt_node(x) for x in raw_nodes if isinstance(x, dict)]
    entities = [x for x in entities if x["id"] and x["name"] and x["type"]]
    bilingual = _load_bilingual_aliases()
    for entity in entities:
        name_lower = entity["name"].lower()
        extras = [
            alias
            for group in bilingual
            if any(pattern and pattern in name_lower for pattern in group["match"])
            for alias in group["aliases"]
        ]
        if extras:
            existing = set(entity["aliases"])
            entity["aliases"] = list(entity["aliases"]) + [a for a in extras if a not in existing]
    buckets: Dict[str, List[Dict[str, Any]]] = {layer: [] for layer in _LAYERS}
    for entity in entities:
        buckets[entity["layer"]].append(entity)
    relations = [_adapt_relation(x) for x in raw_relations if isinstance(x, dict)]
    relations = [x for x in relations if x["source_id"] and x["target_id"] and x["relation_type"]]
    return {
        "document": {
            "file_name": "master_ontology",
            "file_path": str(root),
            "domain": "ENTERPRISE",
        },
        "layer_scope": "spec",
        "entities": buckets,
        "relations": relations,
        "evidence": [],
        "provenance": {"origin": "master_ontology_adapter"},
        "quality": {
            "node_count": len(entities),
            "relation_count": len(relations),
        },
        "warnings": [],
        "assumptions": [
            "nodes.json is canonical; instances.json is not appended to avoid duplicate IDs",
            "master export has no source_cq for relations unless later enriched",
        ],
    }
