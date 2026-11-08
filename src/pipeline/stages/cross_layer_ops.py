"""Cross-layer violation fix extracted from global_optimizer.

Phase 3 LOC reduction (docs/CODE_CLEANUP_CANDIDATES.md §3.1): the Step-5b
cross-layer trio moved verbatim. Single source of truth remains
config/ontology/relationship_rules.yaml — semantics mirror QualityGate
(see quality_gate._check_layer_pair_rules). global_optimizer re-imports
these names, so `go._ALLOWED_CROSS_LAYER` / `go._fix_cross_layer_violations`
access keeps working (locked by tests/extraction/test_extracted_modules.py).
"""
from __future__ import annotations

from typing import Any

from .entity_accessors import _get_entity_id


def _cross_layer_direction_allowed(
    src_layer: str,
    tgt_layer: str,
    layer_pair_rules: dict[str, Any],
) -> bool:
    """Mirror QualityGate._check_layer_pair_rules exactly (same-layer included)."""
    if src_layer == tgt_layer:
        return True
    pair = f"{src_layer}-{tgt_layer}"
    if pair in layer_pair_rules:
        return bool(layer_pair_rules[pair].get("allowed", True))
    rev = f"{tgt_layer}-{src_layer}"
    if rev in layer_pair_rules:
        return bool(layer_pair_rules[rev].get("allowed", True))
    return True


def _load_cross_layer_rules() -> dict[str, Any]:
    import os

    rule_path = os.environ.get(
        "ONTOLOGY_RELATIONSHIP_RULES",
        "config/ontology/relationship_rules.yaml",
    )
    try:
        import yaml

        with open(rule_path, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f) or {}
    except Exception as exc:  # pragma: no cover - config is present in-repo
        print(f"[GlobalOpt] WARNING cross-layer rules unavailable ({exc}); "
              f"falling back to forward-only L0->L1->L2->L3")
        rules = {
            "layer_pair_rules": {
                "L0-L1": {"allowed": True},
                "L1-L2": {"allowed": True},
                "L2-L3": {"allowed": True},
                "L2.5-L3": {"allowed": True},
            }
        }
    return rules


_ALLOWED_CROSS_LAYER: frozenset[tuple[str, str]] = frozenset(
    (src, tgt)
    for src in ("L0", "L1", "L2", "L2.5", "L3")
    for tgt in ("L0", "L1", "L2", "L2.5", "L3")
    if _cross_layer_direction_allowed(
        src, tgt, _load_cross_layer_rules().get("layer_pair_rules") or {}
    )
)


def _fix_cross_layer_violations(
    relationships: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> tuple[int, int]:
    """Fix cross-layer relationship violations by reversing direction.

    For each relationship where (src_layer, tgt_layer) is not in the allowed set:
    - If reversed direction (tgt_layer, src_layer) IS allowed, swap source/target
    - If neither direction is allowed, remove the relationship

    Allowed directions load from config/ontology/relationship_rules.yaml, the
    same source used by QualityGate (see _cross_layer_direction_allowed).

    Returns (reversed_count, removed_count).
    """
    layer_pair_rules = (_load_cross_layer_rules().get("layer_pair_rules") or {})

    # Build layer lookup
    layer_map: dict[str, str] = {}
    for n in nodes:
        if isinstance(n, dict):
            eid = _get_entity_id(n)
            if eid:
                layer_map[eid] = n.get("layer", "")

    reversed_count = 0
    removed_count = 0
    cleaned: list[dict[str, Any]] = []

    for rel in relationships:
        if not isinstance(rel, dict):
            cleaned.append(rel)
            continue

        src = str(rel.get("source") or rel.get("from") or "")
        tgt = str(rel.get("target") or rel.get("to") or "")
        sl = layer_map.get(src, "")
        tl = layer_map.get(tgt, "")

        # Same-layer or missing layer info — keep as-is
        if not sl or not tl or sl == tl:
            cleaned.append(rel)
            continue

        # Check if direction is allowed
        if _cross_layer_direction_allowed(sl, tl, layer_pair_rules):
            cleaned.append(rel)
            continue

        # Try reversing
        if _cross_layer_direction_allowed(tl, sl, layer_pair_rules):
            # Swap source and target
            rel["source"], rel["target"] = tgt, src
            # Also swap from/to if present
            if "from" in rel or "to" in rel:
                rel["from"], rel["to"] = tgt, src
            reversed_count += 1
            cleaned.append(rel)
        else:
            # Neither direction is allowed — remove
            removed_count += 1

    if reversed_count or removed_count:
        print(f"[GlobalOpt] Cross-layer fix: {reversed_count} reversed, "
              f"{removed_count} removed")

    # Replace in-place
    relationships.clear()
    relationships.extend(cleaned)
    return reversed_count, removed_count
