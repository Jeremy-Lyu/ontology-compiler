"""RELATED_TO → semantic relation rewrite policy (G7, B-7, 2026-08-17).

Extracted from ``MainPipeline`` — the RELATED_TO rewrite family is a business
decision (rule-based semantic mapping), not an orchestration side effect, so
it now lives here where it can be unit-tested and reused as a skill core
(relation_rewrite C-1 target). Pure functions; the only class dependency
(``_RELATED_TO_REWRITE_MAP``) is now a module constant.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


RELATED_TO_REWRITE_MAP: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bdepends?\s+on\b", re.I), "DEPENDS_ON"),
    (re.compile(r"\brequires?\b", re.I), "DEPENDS_ON"),
    (re.compile(r"\brelies\s+on\b", re.I), "DEPENDS_ON"),
    (re.compile(r"\bsupports?\b", re.I), "SUPPORTS"),
    (re.compile(r"\benables?\b", re.I), "SUPPORTS"),
    (re.compile(r"\bassists?\b", re.I), "SUPPORTS"),
    (re.compile(r"\bfacilitates?\b", re.I), "SUPPORTS"),
    (re.compile(r"\bderived\s+from\b", re.I), "DERIVED_FROM"),
    (re.compile(r"\bderived\s+by\b", re.I), "DERIVED_FROM"),
    (re.compile(r"\bengineering\s+bom\b.*\bmanufacturing\s+bom\b", re.I), "DERIVED_FROM"),
    (re.compile(r"\boriginates?\s+from\b", re.I), "DERIVED_FROM"),
    (re.compile(r"\btransforms?\s+to\b", re.I), "TRANSFORMS_TO"),
    (re.compile(r"\bpart\s+of\b", re.I), "PART_OF"),
    (re.compile(r"\bcontained\s+in\b", re.I), "PART_OF"),
    (re.compile(r"\blocated\s+in\b", re.I), "LOCATED_IN"),
    (re.compile(r"\bat\s+each\s+plant\b", re.I), "LOCATED_IN"),
    (re.compile(r"\bassociated\s+with\b", re.I), "ASSOCIATED_WITH"),
    (re.compile(r"\brelated\s+to\b", re.I), "ASSOCIATED_WITH"),
)


def build_relation_text_blob(rel: Dict[str, Any]) -> str:
    return " ".join(str(v) for v in rel.values() if isinstance(v, str)).lower()


def collect_related_to_candidates(
    *,
    text_blob: str,
    source_type: str,
    target_type: str,
    source_name: str,
    target_name: str,
) -> list[str]:
    def _has_any(*tokens: str) -> bool:
        return any(token in text_blob for token in tokens)

    candidates: list[str] = []

    if source_type in {"actor", "system"} and target_type in {"system", "process"}:
        candidates.extend(["SUPPORTS", "PARTICIPATES_IN"])
    if source_type in {"data", "asset"} and target_type in {"process", "system"}:
        candidates.extend(["CONSUMES_AT", "TRANSFORMS_TO"])
    if source_type in {"process", "capability"} and target_type in {"process", "capability"}:
        candidates.extend(["DEPENDS_ON", "ASSOCIATED_WITH"])
    if source_type in {"tool", "system"} and target_type in {"process", "task"}:
        candidates.extend(["SUPPORTS", "DEPENDS_ON"])

    if _has_any("depends on", "requires", "relies on", "dependency"):
        candidates.insert(0, "DEPENDS_ON")
    if _has_any("supports", "enables", "assists", "facilitates"):
        candidates.insert(0, "SUPPORTS")
    if _has_any("consumes", "inputs to", "used by"):
        candidates.insert(0, "CONSUMES_AT")
    if _has_any("derived from", "derived by", "engineering bom", "manufacturing bom", "from source"):
        candidates.insert(0, "DERIVED_FROM")
    if _has_any("part of", "belongs to", "contained in"):
        candidates.insert(0, "PART_OF")
    if _has_any("located in", "at each plant", "in plant", "site", "location"):
        candidates.insert(0, "LOCATED_IN")
    if _has_any("meeting", "forum", "workshop"):
        candidates.insert(0, "PARTICIPATES_IN")

    if "tool" in source_name or "tool" in target_name:
        candidates.append("SUPPORTS")
    if "hub" in source_name or "hub" in target_name:
        candidates.append("ASSOCIATED_WITH")
    if "plant" in source_name or "plant" in target_name:
        candidates.append("LOCATED_IN")
    if "inventory" in source_name and "control" in source_name and target_name:
        candidates.append("APPLIES_TO")

    for pattern, replacement in RELATED_TO_REWRITE_MAP:
        if pattern.search(text_blob):
            candidates.insert(0, replacement)
    return candidates


def dedupe_ordered_candidates(candidates: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def finalize_related_to_relation(
    *,
    rel: Dict[str, Any],
    ordered_candidates: list[str],
    source_type: str,
    target_type: str,
    source_name: str,
    target_name: str,
) -> Dict[str, Any]:
    rel = dict(rel)
    rel["rewrite_context"] = {
        "source_type": source_type,
        "target_type": target_type,
        "source_name": source_name,
        "target_name": target_name,
    }
    rel.setdefault("rewrite_candidates", ordered_candidates[:5])
    if ordered_candidates:
        rel["type"] = ordered_candidates[0]
        rel["relation_type"] = ordered_candidates[0]
        rel.setdefault("rewritten_from", "RELATED_TO")
        rel.setdefault("rewrite_reason", ordered_candidates[0])
    return rel


def rewrite_related_to(rel: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestrator: RELATED_TO → semantic relation (pure, no pipeline state)."""
    text_blob = build_relation_text_blob(rel)
    source_type = str(rel.get("source_type", "")).lower().strip()
    target_type = str(rel.get("target_type", "")).lower().strip()
    source_name = str(rel.get("source_name", "")).lower().strip()
    target_name = str(rel.get("target_name", "")).lower().strip()

    candidates = collect_related_to_candidates(
        text_blob=text_blob,
        source_type=source_type,
        target_type=target_type,
        source_name=source_name,
        target_name=target_name,
    )
    ordered = dedupe_ordered_candidates(candidates)
    return finalize_related_to_relation(
        rel=rel,
        ordered_candidates=ordered,
        source_type=source_type,
        target_type=target_type,
        source_name=source_name,
        target_name=target_name,
    )