"""Deterministic entity drift metrics (ROADMAP T4.2).

Compares two packet snapshots without writing files or making model calls.
This is intentionally a signal/report primitive, not an auto-merge policy:
resolution stays an explicit, reversible operator action.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

__all__ = ["entity_drift_report"]


def _entities(packet: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    field = packet.get("entities") or {}
    buckets = field.values() if isinstance(field, dict) else [field]
    out: Dict[str, Dict[str, Any]] = {}
    for bucket in buckets:
        if not isinstance(bucket, list):
            continue
        for ent in bucket:
            if isinstance(ent, dict) and ent.get("id"):
                out[str(ent["id"])] = ent
    return out


def _fingerprint(ent: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(ent.get("name") or "").strip(),
        str(ent.get("type") or "").strip(),
        str(ent.get("layer") or "").strip(),
        str(ent.get("version") or "").strip(),
        str(ent.get("description") or "").strip(),
    )


def entity_drift_report(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    """Report added/removed/changed/unchanged entities between snapshots."""
    old, new = _entities(before), _entities(after)
    old_ids, new_ids = set(old), set(new)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    changed = sorted(
        eid for eid in old_ids & new_ids if _fingerprint(old[eid]) != _fingerprint(new[eid])
    )
    unchanged = sorted((old_ids & new_ids) - set(changed))
    total = max(len(old_ids), len(new_ids), 1)
    return {
        "before_count": len(old_ids),
        "after_count": len(new_ids),
        "added_ids": added,
        "removed_ids": removed,
        "changed_ids": changed,
        "unchanged_ids": unchanged,
        "changed_ratio": round((len(added) + len(removed) + len(changed)) / total, 4),
    }
