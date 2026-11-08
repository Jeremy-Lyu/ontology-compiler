"""Shared entity field accessors extracted from global_optimizer.

Phase 3 LOC reduction (docs/CODE_CLEANUP_CANDIDATES.md §3.1): tiny
entity-dict accessors moved verbatim so that sibling modules (e.g.
cross_layer_ops) can use them without importing global_optimizer
(avoiding a package-internal import cycle). global_optimizer re-imports
these names, so `go._get_entity_id` style access keeps working.
"""
from __future__ import annotations

from typing import Any


def _normalize_name(name: str) -> str:
    """Normalize entity name for cross-layer duplicate matching."""
    if not name or not isinstance(name, str):
        return ""
    n = name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(n.split())


def _get_entity_id(node: dict[str, Any]) -> str:
    return str(node.get("meta_id") or node.get("entity_id") or node.get("id") or "")


def _get_entity_name(node: dict[str, Any]) -> str:
    return str(node.get("name") or node.get("meta_name") or node.get("scenario_name") or "")


def _set_entity_id(node: dict[str, Any], new_id: str) -> None:
    """Update whichever ID-bearing field(s) the entity already uses."""
    touched = False
    for key in ("meta_id", "entity_id"):
        if key in node:
            node[key] = new_id
            touched = True
    if not touched and "id" in node:
        node["id"] = new_id
