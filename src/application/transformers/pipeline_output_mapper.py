"""Pipeline output mapper (B1, 2026-08-15).

Extracted from ``MainPipeline._collect_pipeline_output`` so the schema
normalization (onto the OutputValidator / ontology_packet aliases) is a pure,
independently testable unit instead of a 130-line inline block in the
orchestrator.

Split responsibilities:
- ``MainPipeline`` keeps the I/O: read run_manifest / merge master, fall back
  to per-stage mixing, and read context metadata.
- This mapper performs the pure transformations: entity/relation field aliases
  and the final output-view assembly.

These aliases are intentionally the same contract the quality gate and
ontpack validator expect (entity_id/canonical_name/entity_type OR
meta_id/meta_name/meta_class; relations from/to -> source_id/target_id).
"""

from __future__ import annotations

from typing import Any, Dict, List


def normalize_pipeline_entities(entities: List[Any]) -> List[Any]:
    """Normalise entity fields onto what OutputValidator and the
    ontology_packet schema expect: preserve the actual keys
    (entity_id/canonical_name/entity_type or meta_id/meta_name/meta_class)
    while also exposing the conventional id/name/type aliases."""
    normalized: List[Any] = []
    for entity in entities:
        if not isinstance(entity, dict):
            normalized.append(entity)
            continue
        out = dict(entity)
        out.setdefault("id", out.get("entity_id") or out.get("meta_id") or "")
        out.setdefault("name", out.get("canonical_name") or out.get("meta_name") or out.get("name") or "")
        out.setdefault("type", out.get("entity_type") or out.get("meta_class") or "")
        out.setdefault("layer", out.get("layer", ""))
        normalized.append(out)
    return normalized


def normalize_pipeline_relationships(relationships: List[Any]) -> List[Any]:
    """Map stage-named node/relations onto the fields OutputValidator and the
    ontology_packet schema expect (entities / relationships), and normalise
    relationship endpoint keys (from/to -> source_id/target_id)."""
    normalized: List[Any] = []
    for rel in relationships:
        if not isinstance(rel, dict):
            normalized.append(rel)
            continue
        out = dict(rel)
        if "source_id" not in out and ("from" in out or "source" in out):
            out["source_id"] = out.get("from") or out.get("source")
        if "target_id" not in out and ("to" in out or "target" in out):
            out["target_id"] = out.get("to") or out.get("target")
        if "type" in out and "relation_type" not in out and not isinstance(out.get("type"), dict):
            out.setdefault("relation_type", out["type"])
        normalized.append(out)
    return normalized


def assemble_pipeline_output_view(
    *,
    source_path: str,
    nodes: List[Any],
    instances: List[Any],
    relationships: List[Any],
    metadata: Dict[str, Any],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the output-view dict used by the quality gate / post-execution
    validation. ``nodes``/``instances``/``relationships``/``metadata`` are the
    authoritative payloads (already selected by the caller); this function only
    normalizes aliases and shapes the view."""
    entities = nodes or instances
    normalized_entities = normalize_pipeline_entities(entities)
    normalized_relationships = normalize_pipeline_relationships(relationships)

    return {
        "document": metadata.get("document_path") or source_path or "",
        "entities": normalized_entities,
        "relations": normalized_relationships,
        "relationships": normalized_relationships,
        "nodes": nodes,
        "instances": instances,
        "metadata": metadata,
        "stats": {
            "entity_count": stats.get("entity_count", 0),
            "relation_count": stats.get("relation_count", 0),
        },
    }