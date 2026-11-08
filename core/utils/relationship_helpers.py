from __future__ import annotations

from typing import Any, Iterable


RELATIONSHIP_PROVENANCE_VALUES = frozenset({
    "same_as_detection",
    "cross_file_association",
    "manual_fallback",
    "run_pipeline_orchestrator",
})


def entity_to_dict(entity: Any) -> dict[str, Any]:
    if isinstance(entity, dict):
        return entity
    if hasattr(entity, "model_dump"):
        data = entity.model_dump()
        if "evidence" in data and data["evidence"]:
            if hasattr(data["evidence"], "model_dump"):
                data["evidence"] = data["evidence"].model_dump()
        return data
    if hasattr(entity, "to_dict"):
        return entity.to_dict()
    return {}


def collect_entity_ids(entities: Iterable[Any]) -> set[str]:
    ids: set[str] = set()
    for entity in entities:
        data = entity_to_dict(entity)
        for key in ("meta_id", "id", "identifier"):
            value = data.get(key)
            if value:
                ids.add(str(value))
    return ids


def relationship_endpoints(rel: Any) -> tuple[str, str]:
    if not isinstance(rel, dict):
        rel = entity_to_dict(rel)

    from_id = str(rel.get("from") or "").strip()
    to_id = str(rel.get("to") or "").strip()
    source_id = str(rel.get("source") or "").strip()
    target_id = str(rel.get("target") or "").strip()

    if source_id in RELATIONSHIP_PROVENANCE_VALUES:
        source_id = from_id
    elif not source_id:
        source_id = from_id

    if not target_id:
        target_id = to_id

    return source_id, target_id


def is_valid_relationship(rel: Any, entity_ids: set[str]) -> bool:
    source_id, target_id = relationship_endpoints(rel)
    if not source_id or not target_id:
        return False
    if source_id == target_id:
        return False
    if source_id not in entity_ids and target_id not in entity_ids:
        return False
    return True


def is_valid_relationship_relaxed(rel: Any, entity_ids: set[str]) -> bool:
    source_id, target_id = relationship_endpoints(rel)
    if not source_id or not target_id:
        return False
    if source_id == target_id:
        return False
    return True


def normalize_relationship_dict(rel: Any) -> dict[str, Any]:
    data = entity_to_dict(rel)
    source_id, target_id = relationship_endpoints(data)
    normalized = dict(data)
    normalized["source"] = source_id
    normalized["target"] = target_id
    normalized.setdefault(
        "type",
        data.get("type") or data.get("rel_class") or data.get("predicate") or "related_to",
    )
    return normalized


def prune_dangling_relationships(
    relationships: Iterable[Any],
    entity_ids: set[str],
    allow_dangling: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    dropped = 0
    validator = is_valid_relationship_relaxed if allow_dangling else is_valid_relationship
    for rel in relationships:
        if validator(rel, entity_ids):
            kept.append(normalize_relationship_dict(rel))
        else:
            dropped += 1
    return kept, dropped
