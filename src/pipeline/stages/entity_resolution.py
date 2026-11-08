"""Multi-Strategy Entity Resolution (MSER) Module.

Resolves entities that refer to the same real-world object but are extracted
with different names across documents/sources (e.g., "LUQIAO" vs "LUQIAO PLANT"
vs "VCTZ").

Strategies (applied in priority order):
  A. Alias dictionary — exact match against curated alias table
  B. Substring/containment — one name is a substring of the other
  C. Fuzzy string matching — rapidfuzz (ratio, partial_ratio, token_sort)
  D. Context/neighbor matching — shared relationship targets
  E. (Future) LLM semantic disambiguation

Usage:
    from src.pipeline.stages.entity_resolution import resolve_entities_mser

    consolidated_nodes, id_remap, stats = resolve_entities_mser(
        nodes, relationships, alias_path="config/entity_aliases.yaml"
    )
"""

from __future__ import annotations

import os
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("PIPELINE.MSER")

# Try importing rapidfuzz (optional, graceful fallback)
try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    logger.info("rapidfuzz not installed, fuzzy matching disabled (pip install rapidfuzz)")

# Try importing yaml for alias config
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Result of evaluating whether two entities are the same."""
    score: float          # Combined score 0.0-1.0
    strategy: str         # Which strategy matched ("alias", "substring", "fuzzy", "context", "none")
    should_merge: bool     # Auto-merge
    needs_review: bool     # Flag for manual review
    reason: str = ""       # Human-readable explanation


@dataclass
class MSERStats:
    """Statistics for MSER processing."""
    alias_matches: int = 0
    substring_matches: int = 0
    fuzzy_matches: int = 0
    context_matches: int = 0
    review_flagged: int = 0
    total_compared: int = 0
    total_merged: int = 0

    def __str__(self) -> str:
        return (f"compared={self.total_compared}, merged={self.total_merged} "
                f"(alias={self.alias_matches}, substring={self.substring_matches}, "
                f"fuzzy={self.fuzzy_matches}, context={self.context_matches}, "
                f"review={self.review_flagged})")


# ---------------------------------------------------------------------------
# Strategy A: Alias Dictionary
# ---------------------------------------------------------------------------

def load_alias_map(path: str) -> dict[str, str]:
    """Load alias dictionary from YAML config file.

    Returns a mapping: normalized_variant_name → canonical_name
    """
    if not _HAS_YAML or not os.path.exists(path):
        logger.warning(f"Alias config not found or PyYAML not installed: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    alias_map: dict[str, str] = {}
    for entry in data.get("aliases", []):
        canonical = _normalize_text(entry["canonical"])
        if not canonical:
            continue
        # Map canonical to itself
        alias_map[canonical] = canonical
        # Map all variants to canonical
        for variant in entry.get("variants", []):
            v = _normalize_text(variant)
            if v:
                alias_map[v] = canonical

    logger.info(f"Loaded {len(alias_map)} alias entries from {path}")
    return alias_map


def _match_by_alias(name: str, alias_map: dict[str, str]) -> str | None:
    """Check if name matches any alias. Returns canonical name or None."""
    key = _normalize_text(name)
    return alias_map.get(key)


# ---------------------------------------------------------------------------
# Strategy B: Substring / Containment Matching
# ---------------------------------------------------------------------------

def _match_by_substring(name1: str, name2: str, min_len: int = 4) -> bool:
    """Check if one name is a substring of the other (both >= min_len chars).

    Handles cases like:
      "LUQIAO" vs "LUQIAO PLANT"  → True (LUQIAO ⊂ LUQIAO PLANT)
      "EBOM" vs "EBOM Structure"  → True
      "GOB" vs "Global Order BOM" → False (different tokens)
    """
    n1 = _normalize_text(name1)
    n2 = _normalize_text(name2)
    if len(n1) < min_len or len(n2) < min_len:
        return False
    if n1 == n2:
        return False  # Exact match handled elsewhere
    # Shorter name must be a substring of the longer one
    if len(n1) <= len(n2):
        return n1 in n2
    return n2 in n1


# ---------------------------------------------------------------------------
# Strategy C: Fuzzy String Matching (rapidfuzz)
# ---------------------------------------------------------------------------

def _match_by_fuzzy(name1: str, name2: str, threshold: float = 0.85) -> float:
    """Calculate fuzzy similarity score using rapidfuzz.

    Returns the maximum score across multiple fuzzy matching algorithms:
    - ratio: Global edit-distance similarity
    - partial_ratio: Best matching substring (handles "LUQIAO" vs "LUQIAO PLANT")
    - token_sort_ratio: Word-order-independent matching

    Returns 0.0 if rapidfuzz is not installed.
    """
    if not _HAS_RAPIDFUZZ:
        return 0.0

    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0

    scores = [
        _fuzz.ratio(n1, n2) / 100.0,
        _fuzz.partial_ratio(n1, n2) / 100.0,
        _fuzz.token_sort_ratio(n1, n2) / 100.0,
    ]
    return max(scores)


# ---------------------------------------------------------------------------
# Strategy D: Context / Neighbor Matching
# ---------------------------------------------------------------------------

def _build_neighbor_map(
    relationships: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """Build entity_id → set(neighbor_ids) map from relationships."""
    neighbors: dict[str, set[str]] = defaultdict(set)
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        src = str(rel.get("source") or rel.get("from") or "")
        tgt = str(rel.get("target") or rel.get("to") or "")
        if src and tgt:
            neighbors[src].add(tgt)
            neighbors[tgt].add(src)
    return dict(neighbors)


def _match_by_context(
    id1: str, id2: str,
    neighbor_map: dict[str, set[str]],
) -> float:
    """Calculate context similarity based on shared relationship neighbors.

    If two entities share many relationship partners, they might be the same
    entity extracted from different sources.

    Returns Jaccard similarity of neighbor sets (0.0-1.0).
    """
    n1 = neighbor_map.get(id1, set())
    n2 = neighbor_map.get(id2, set())
    if not n1 or not n2:
        return 0.0
    # Exclude each other from neighbor sets (they might reference each other)
    n1 = n1 - {id2}
    n2 = n2 - {id1}
    if not n1 or not n2:
        return 0.0
    intersection = n1 & n2
    union = n1 | n2
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Combined Evaluation
# ---------------------------------------------------------------------------

# Thresholds
AUTO_MERGE_THRESHOLD = 0.72   # Auto-merge when score >= 0.72
REVIEW_THRESHOLD = 0.58        # Flag for review when 0.58 <= score < 0.72
FUZZY_THRESHOLD = 0.75         # Fuzzy score alone must be >= this to auto-merge
CONTEXT_THRESHOLD = 0.45       # Context score alone must be >= this to auto-merge

# Class-based boost: when both entities have the same meta_class, boost the combined score
CLASS_MATCH_BOOST = 0.05
# Layer-based boost: when both entities are at the same layer
LAYER_MATCH_BOOST = 0.02


def _token_overlap(name1: str, name2: str) -> float:
    """Calculate token-level overlap (Jaccard) between two entity names.

    Splits names into tokens (by spaces, underscores, hyphens, camelCase),
    then computes Jaccard similarity of token sets.
    """
    import re
    def _tokenize(s: str) -> set[str]:
        s = s.lower().strip()
        # Split on non-alphanumeric, plus camelCase boundaries
        tokens = set(re.findall(r'[a-z]+|[0-9]+', s))
        # Also split camelCase within tokens
        expanded = set()
        for t in tokens:
            expanded.update(re.findall(r'[a-z]+|[0-9]+', t))
        return expanded if expanded else tokens

    t1 = _tokenize(name1)
    t2 = _tokenize(name2)
    if not t1 or not t2:
        return 0.0
    intersection = t1 & t2
    union = t1 | t2
    return len(intersection) / len(union) if union else 0.0


def _evaluate_match(
    name1: str, name2: str,
    id1: str, id2: str,
    meta_class1: str, meta_class2: str,
    alias_map: dict[str, str],
    neighbor_map: dict[str, set[str]],
) -> MatchResult:
    """Evaluate whether two entities are the same using all strategies."""

    # --- Strategy A: Alias dictionary ---
    c1 = _match_by_alias(name1, alias_map)
    c2 = _match_by_alias(name2, alias_map)
    if c1 and c2 and c1 == c2:
        return MatchResult(1.0, "alias", True, False,
                           f"Alias match: '{name1}' and '{name2}' → '{c1}'")

    # --- Strategy B: Substring matching ---
    if _match_by_substring(name1, name2):
        return MatchResult(0.90, "substring", True, False,
                           f"Substring: '{name1}' ⊂/⊃ '{name2}'")

    # --- Strategy C: Fuzzy string matching ---
    fuzzy_score = _match_by_fuzzy(name1, name2)

    # --- Strategy D: Context/neighbor matching ---
    context_score = _match_by_context(id1, id2, neighbor_map)

    # --- Strategy E: Token overlap ---
    token_score = _token_overlap(name1, name2)

    # --- Class/Layer match boosts ---
    class_match = meta_class1.strip().lower() == meta_class2.strip().lower() and meta_class1
    class_boost = CLASS_MATCH_BOOST if class_match else 0.0

    # --- Combined scoring ---
    # Weighted combination of all signals (project memory 硬约束公式: fuzzy*0.5 + context*0.3 + token*0.2 + class_boost)
    # Fuzzy is the primary signal, context and token are supporting
    combined = 0.0
    strategy_tag = "combined"  # 默认标签，下面细分

    if fuzzy_score >= FUZZY_THRESHOLD:
        # High fuzzy confidence — auto-merge with class boost
        combined = fuzzy_score + class_boost
        strategy_tag = "fuzzy"
    elif fuzzy_score >= 0.65:
        # Moderate fuzzy — combine with context and token signals (标准加权公式分支)
        combined = fuzzy_score * 0.5 + context_score * 0.3 + token_score * 0.2 + class_boost
        strategy_tag = "fuzzy+context" if context_score >= CONTEXT_THRESHOLD else "fuzzy"
    elif context_score >= CONTEXT_THRESHOLD and token_score >= 0.5:
        # Strong context + token overlap — likely same entity (context-主导分支)
        combined = context_score * 0.5 + token_score * 0.3 + fuzzy_score * 0.2 + class_boost
        strategy_tag = "fuzzy+context"
    elif token_score >= 0.6 and class_match:
        # High token overlap + same class — likely same entity
        combined = token_score * 0.6 + fuzzy_score * 0.4 + class_boost
        strategy_tag = "combined"
    else:
        # Low confidence — just use fuzzy as best signal
        combined = fuzzy_score + class_boost
        strategy_tag = "none" if combined < REVIEW_THRESHOLD else "combined"

    # Clamp to [0, 1]
    combined = min(1.0, max(0.0, combined))

    if combined >= AUTO_MERGE_THRESHOLD:
        return MatchResult(combined, strategy_tag, True, False,
                           f"Auto-merge: strategy={strategy_tag}, fuzzy={fuzzy_score:.2f}, context={context_score:.2f}, "
                           f"token={token_score:.2f}, class_boost={class_boost:.2f}")
    elif combined >= REVIEW_THRESHOLD:
        return MatchResult(combined, strategy_tag, False, True,
                           f"Review: strategy={strategy_tag}, fuzzy={fuzzy_score:.2f}, context={context_score:.2f}, "
                           f"token={token_score:.2f}, class_boost={class_boost:.2f}")

    return MatchResult(combined, "none", False, False, "No match")


_CANONICAL_ANCHOR_NAMES = {
    "bom", "ebom", "mbom",
    "inventory", "cycle stock", "safety stock",
    "master data", "vsm", "production", "benchmark", "atacq",
}


def _is_canonical_anchor(node: dict[str, Any]) -> bool:
    """True for canonical domain anchors that extract's safety net re-injects.

    These are the golden-eval reference points (BOM/EBOM/MBOM, Inventory,
    ...). Folding them into older, differently-named master nodes via fuzzy or
    substring matching would both destroy the golden anchor and corrupt this
    run's fresh relation endpoints, so they are protected from such merges.
    """
    if node.get("is_canonical") or node.get("preserve_on_merge"):
        return True
    return _normalize_text(_get_entity_name(node)) in _CANONICAL_ANCHOR_NAMES


def _guard_canonical_merge(n1: dict[str, Any], n2: dict[str, Any]) -> bool:
    """Return True when a canonical anchor must NOT be merged with a differently
    named node (e.g. BOM should not be folded into EBOM, nor MBOM into
    "MBOM Art Fg Pos Pa Structure"). Exact-name duplicates still merge."""
    if not (_is_canonical_anchor(n1) or _is_canonical_anchor(n2)):
        return False
    name1 = _normalize_text(_get_entity_name(n1))
    name2 = _normalize_text(_get_entity_name(n2))
    return bool(name1) and bool(name2) and name1 != name2


# ---------------------------------------------------------------------------
# Blocking: reduce O(n²) comparisons
# ---------------------------------------------------------------------------

def _build_blocks(
    nodes: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group entities into blocks for efficient pairwise comparison.

    Block keys (two-level blocking to reduce O(n²) pairs):
    - Level 1: "{layer}|{meta_class}" — same layer + same class
    - Level 2: "{layer}|{meta_class}|{name_prefix}" — first 3 chars of name

    Entities in the same Level 1 block but different Level 2 blocks
    are only compared if they share a common alias or substring pattern.
    This reduces comparison pairs from O(n²) to ~O(n * k) where k = avg block size.
    """
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        layer = _get_field(node, "layer", "UNKNOWN")
        cls = _get_field(node, "meta_class", "").strip().lower()
        if cls:
            name = _get_entity_name(node)
            # Use first 3 alphanumeric chars as secondary key
            prefix = ""
            for ch in name.lower():
                if ch.isalnum():
                    prefix += ch
                if len(prefix) >= 3:
                    break
            if not prefix:
                prefix = "_"
            blocks[f"{layer}|{cls}|{prefix}"].append(node)
    return dict(blocks)


# ---------------------------------------------------------------------------
# Main Resolution Function
# ---------------------------------------------------------------------------

def resolve_entities_mser(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    alias_path: str = "config/entity_aliases.yaml",
    existing_remap: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str], MSERStats]:
    """Run multi-strategy entity resolution on a list of nodes.

    Args:
        nodes: List of entity nodes to resolve.
        relationships: List of relationships (for context matching).
        alias_path: Path to alias YAML config file.
        existing_remap: Existing ID remap from previous dedup steps.

    Returns:
        (consolidated_nodes, id_remap, stats)
    """
    stats = MSERStats()
    id_remap = dict(existing_remap or {})

    # Load alias dictionary
    alias_map = load_alias_map(alias_path) if alias_path else {}

    # Build neighbor map for context matching
    neighbor_map = _build_neighbor_map(relationships)

    # Build blocks (two-level: layer|class|name_prefix)
    blocks = _build_blocks(nodes)

    # Group blocks by layer|class for cross-block alias/substring matching
    lc_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block_key, block_nodes in blocks.items():
        parts = block_key.split("|")
        if len(parts) >= 2:
            lc_key = f"{parts[0]}|{parts[1]}"
            lc_groups[lc_key].extend(block_nodes)

    merged_ids: set[str] = set()

    # Phase 1: Compare within each name-prefix block (fast, covers most cases)
    for block_key, block_nodes in blocks.items():
        if len(block_nodes) < 2:
            continue

        block_nodes.sort(key=lambda n: _completeness_score(n), reverse=True)

        for i, n1 in enumerate(block_nodes):
            id1 = _get_entity_id(n1)
            name1 = _get_entity_name(n1)
            cls1 = _get_field(n1, "meta_class", "")
            if id1 in merged_ids:
                continue

            for n2 in block_nodes[i + 1:]:
                id2 = _get_entity_id(n2)
                if id2 in merged_ids or id1 == id2:
                    continue

                name2 = _get_entity_name(n2)
                cls2 = _get_field(n2, "meta_class", "")

                stats.total_compared += 1

                result = _evaluate_match(
                    name1, name2, id1, id2,
                    cls1, cls2,
                    alias_map, neighbor_map,
                )

                if result.should_merge and not _guard_canonical_merge(n1, n2):
                    if id2 not in id_remap:
                        id_remap[id2] = id1
                        _merge_fields(n1, n2)
                        merged_ids.add(id2)
                        stats.total_merged += 1

                        if result.strategy == "alias":
                            stats.alias_matches += 1
                        elif result.strategy == "substring":
                            stats.substring_matches += 1
                        elif result.strategy == "fuzzy":
                            stats.fuzzy_matches += 1
                        elif result.strategy == "fuzzy+context":
                            stats.context_matches += 1

                        logger.debug(f"[MSER] Merged '{name2}' → '{name1}' "
                                     f"(strategy={result.strategy}, score={result.score:.2f})")

                elif result.needs_review:
                    n2.setdefault("review_status", "entity_match_review")
                    n2.setdefault("review_reason",
                                  f"Possible match with '{name1}' "
                                  f"(strategy={result.strategy}, score={result.score:.2f})")
                    stats.review_flagged += 1

    # Phase 2: Cross-block alias+substring matching within same layer|class
    # Only check alias and substring strategies (not fuzzy) for efficiency
    for lc_key, lc_nodes in lc_groups.items():
        if len(lc_nodes) < 2:
            continue

        # Only check nodes not already merged
        active_nodes = [n for n in lc_nodes if _get_entity_id(n) not in merged_ids]
        if len(active_nodes) < 2:
            continue

        active_nodes.sort(key=lambda n: _completeness_score(n), reverse=True)

        for i, n1 in enumerate(active_nodes):
            id1 = _get_entity_id(n1)
            name1 = _get_entity_name(n1)
            cls1 = _get_field(n1, "meta_class", "")
            if id1 in merged_ids:
                continue

            for n2 in active_nodes[i + 1:]:
                id2 = _get_entity_id(n2)
                if id2 in merged_ids or id1 == id2:
                    continue

                name2 = _get_entity_name(n2)
                cls2 = _get_field(n2, "meta_class", "")

                # Skip if same block (already compared in Phase 1)
                prefix1 = prefix2 = ""
                for ch in name1.lower():
                    if ch.isalnum():
                        prefix1 += ch
                    if len(prefix1) >= 3:
                        break
                for ch in name2.lower():
                    if ch.isalnum():
                        prefix2 += ch
                    if len(prefix2) >= 3:
                        break
                if prefix1 == prefix2:
                    continue  # Already compared

                # Only do alias + substring check (fast)
                stats.total_compared += 1

                result = _evaluate_match(
                    name1, name2, id1, id2,
                    cls1, cls2,
                    alias_map, neighbor_map,
                )

                if result.should_merge and not _guard_canonical_merge(n1, n2):
                    if id2 not in id_remap:
                        id_remap[id2] = id1
                        _merge_fields(n1, n2)
                        merged_ids.add(id2)
                        stats.total_merged += 1

                        if result.strategy == "alias":
                            stats.alias_matches += 1
                        elif result.strategy == "substring":
                            stats.substring_matches += 1
                        elif result.strategy == "fuzzy":
                            stats.fuzzy_matches += 1
                        elif result.strategy == "fuzzy+context":
                            stats.context_matches += 1

                        logger.debug(f"[MSER] Cross-block merged '{name2}' → '{name1}' "
                                     f"(strategy={result.strategy}, score={result.score:.2f})")

                elif result.needs_review:
                    n2.setdefault("review_status", "entity_match_review")
                    n2.setdefault("review_reason",
                                  f"Possible match with '{name1}' "
                                  f"(strategy={result.strategy}, score={result.score:.2f})")
                    stats.review_flagged += 1

    # Build survivor list
    survivors = [n for n in nodes if isinstance(n, dict) and _get_entity_id(n) not in merged_ids]

    if stats.total_merged > 0:
        print(f"[GlobalOpt] MSER entity resolution: {stats}")

    return survivors, id_remap, stats


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove extra spaces."""
    if not text:
        return ""
    return " ".join(str(text).strip().lower().replace("_", " ").replace("-", " ").split())


def _get_entity_id(node: dict[str, Any]) -> str:
    """Extract entity ID from node."""
    return str(node.get("meta_id") or node.get("entity_id") or node.get("id") or "")


def _get_entity_name(node: dict[str, Any]) -> str:
    """Extract entity name from node."""
    return str(node.get("meta_name") or node.get("name") or node.get("entity_name") or "")


def _get_field(node: dict[str, Any], field: str, default: str = "") -> str:
    """Safely get a string field from node."""
    val = node.get(field)
    return str(val) if val is not None else default


def _completeness_score(node: dict[str, Any]) -> float:
    """Calculate data completeness score (0-1) for a node."""
    fields = ["meta_name", "meta_class", "layer", "parent_id", "parent_class",
              "attributes", "description", "provenance"]
    filled = sum(1 for f in fields if node.get(f))
    return filled / len(fields)


def _merge_fields(winner: dict[str, Any], loser: dict[str, Any]) -> None:
    """Merge fields from loser into winner (don't overwrite existing values)."""
    for key, value in loser.items():
        if value and (key not in winner or not winner[key]):
            winner[key] = value
    # Merge aliases list
    w_aliases = winner.get("aliases", [])
    l_aliases = loser.get("aliases", [])
    if isinstance(w_aliases, list) and isinstance(l_aliases, list):
        combined = list(set(w_aliases + l_aliases))
        winner["aliases"] = combined
    elif isinstance(l_aliases, list):
        winner["aliases"] = l_aliases


# ---------------------------------------------------------------------------
# Strategy E: LLM Semantic Disambiguation
# ---------------------------------------------------------------------------

def _llm_review_match(
    entity_a: dict[str, Any],
    entity_b: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use LLM to decide whether two entities should be merged.

    Args:
        entity_a: First entity dict.
        entity_b: Second entity dict.
        context: Optional additional context (e.g., relationship neighbors).

    Returns:
        dict with keys: should_merge (bool), confidence (float), reason (str),
                       strategy (str = "llm_review")
    """
    try:
        from src.application.bootstrap.llm_gateway import get_llm_gateway

        name_a = entity_a.get("meta_name") or entity_a.get("name") or "Unknown"
        name_b = entity_b.get("meta_name") or entity_b.get("name") or "Unknown"
        class_a = entity_a.get("meta_class") or entity_a.get("entity_type") or ""
        class_b = entity_b.get("meta_class") or entity_b.get("entity_type") or ""
        layer_a = entity_a.get("layer", "")
        layer_b = entity_b.get("layer", "")
        desc_a = entity_a.get("description", "")
        desc_b = entity_b.get("description", "")

        prompt = f"""You are an ontology entity resolution expert. Analyze the following two entities and determine if they refer to the SAME real-world object or DIFFERENT objects.

Entity A:
- Name: {name_a}
- Class: {class_a}
- Layer: {layer_a}
- Description: {desc_a}

Entity B:
- Name: {name_b}
- Class: {class_b}
- Layer: {layer_b}
- Description: {desc_b}

Consider:
1. Do the names refer to the same thing (e.g., "LUQIAO" vs "LUQIAO PLANT" or "VCTZ")?
2. Are the classes compatible (e.g., both are "Process" or "System")?
3. Do the descriptions indicate the same purpose?
4. Could these be different views/abbreviations of the same entity from different systems?

Respond with ONLY a JSON object in this format:
{{"should_merge": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        # C-1/C-5: call through the unified gateway with COST-AWARE ROUTING.
        # Entity resolution disambiguation is CONFLICT_RESOLUTION (HEAVY tier
        # when conflict markers are present or the prompt is long). The routed
        # model tier is cheaper for trivial cases and escalates to HEAVY only
        # when the prompt contains ambiguity signals. Fall back to direct Dify
        # call only if gateway is unavailable (isolated test environments).
        answer_text = ""
        try:
            gateway = get_llm_gateway()
            outcome = gateway.execute_routed(
                prompt,
                task_kind="conflict_resolution",
                has_conflict=True,
                entity_count_hint=2,
                temperature_override=0.0,
                stage="entity_resolution.llm_review",
            )
            answer_text = outcome.text if outcome.success else ""
        except Exception:
            from src.application.rag.infrastructure.dify_client import ask_dify_kb_qa, get_dify_answer_text
            response = ask_dify_kb_qa(query=prompt)
            answer_text = get_dify_answer_text(response)

        if not answer_text:
            return {
                "should_merge": False,
                "confidence": 0.0,
                "reason": "LLM returned empty response",
                "strategy": "llm_review_empty",
            }

        import re
        import json
        
        # Try multiple JSON extraction strategies
        result = None
        
        # Strategy 1: Try to find JSON with balanced braces
        # Find the first { and match with }
        start_idx = answer_text.find('{')
        if start_idx != -1:
            # Search for matching closing brace
            depth = 0
            end_idx = -1
            for i in range(start_idx, len(answer_text)):
                if answer_text[i] == '{':
                    depth += 1
                elif answer_text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
            
            if end_idx != -1:
                try:
                    json_str = answer_text[start_idx:end_idx]
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # Strategy 2: If above failed, try simpler regex
        if result is None:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', answer_text)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        if result is None:
            # Fallback: try to parse boolean from text
            # Look for explicit true/false indicators
            answer_lower = answer_text.lower()
            
            # Check for explicit "should_merge" in text
            if 'should_merge' in answer_lower:
                # Try to find the value near "should_merge"
                sm_match = re.search(r'should_merge[:\s]+(true|false)', answer_lower)
                if sm_match:
                    should_merge = sm_match.group(1) == 'true'
                else:
                    should_merge = 'not merge' not in answer_lower and 'different' in answer_lower
                    should_merge = not should_merge  # "different" means DON'T merge
            else:
                # Simple heuristic: check for keywords
                should_merge = False
                if 'same entity' in answer_lower or 'same object' in answer_lower:
                    should_merge = True
                elif 'different entity' in answer_lower or 'different object' in answer_lower:
                    should_merge = False
            
            return {
                "should_merge": should_merge,
                "confidence": 0.5,
                "reason": f"LLM response parsed from text: {answer_text[:200]}",
                "strategy": "llm_review_fallback",
            }
        
        # Parse the result safely
        should_merge = bool(result.get("should_merge", False))
        confidence = float(result.get("confidence", 0.5))
        reason = str(result.get("reason", ""))
        
        # Safety check: LLM often has contradictions between should_merge and reason
        # Use reason as the primary signal when there's a conflict
        if reason:
            reason_lower = reason.lower()
            
            # Check for negated positive patterns (e.g., "not the same", "no indication of same")
            # These should be treated as negative, not positive
            negation_patterns = [
                "not the same", "not same", "not the same entity", "not the same object",
                "no indication", "no sign of", "no evidence of",
                "doesn't indicate", "does not indicate", 
                "not refer to the same", "not refers to the same",
                "not likely the same", "not probably the same",
                "cannot be considered the same",
            ]
            
            # First, check for strong negative indicators
            negative_indicators = [
                "different", "distinct", "unrelated", 
                "no overlap", "no similarity", "no connection",
                "completely different", "fundamentally different",
                "separate", "independent",
                "different purposes", "different concepts", "different objects",
                "different entity", "different processes", "different systems",
                "different scopes", "different aspects",
                "not merge", "should not merge", "cannot be merged",
                "indicate different", "represent different",
                "serve different", "different functions",
                "no semantic overlap", "no contextual overlap",
                "different concepts entirely",
                "no indication of being", "indication of different",
            ]
            
            # Strong positive indicators: clearly indicates same entity
            # Exclude patterns that could be negated (e.g., "not the same")
            positive_indicators = [
                "same entity", "same object", "should merge", "merge them",
                "same real-world", "same underlying",
                "refer to the same", "refers to the same",
                "likely the same", "probably the same",
                "can be merged", "should be merged",
                "different names for the same",
                "abbreviation of the same",
                "another name for the same",
                "complementary aspects of the same",
                "same underlying concept", "same real-world object",
                "could be the same", "might be the same",
                "likely refers to the same",
            ]
            
            # Remove negated patterns from positive indicators
            has_negative = any(kw in reason_lower for kw in negative_indicators)
            has_positive_raw = any(kw in reason_lower for kw in positive_indicators)
            
            # Check if positive indicator is negated
            has_negated_positive = any(
                pattern in reason_lower and 
                any(neg in reason_lower for neg in ["not ", "no ", "never "])
                for pattern in positive_indicators
                if pattern in reason_lower
            )
            
            # Also check explicit negation patterns
            has_explicit_negation = any(p in reason_lower for p in negation_patterns)
            
            # Final determination
            has_positive = has_positive_raw and not has_negated_positive and not has_explicit_negation
            
            if has_negative and not has_positive:
                # Clear negative - override to False
                if should_merge:
                    logger.warning(f"LLM contradiction: should_merge=True but reason indicates difference. "
                                 f"Overriding to False. Reason: {reason[:150]}")
                should_merge = False
            elif has_positive and not has_negative:
                # Clear positive - override to True
                if not should_merge:
                    logger.warning(f"LLM contradiction: should_merge=False but reason indicates sameness. "
                                 f"Overriding to True. Reason: {reason[:150]}")
                should_merge = True
            elif has_negative and has_positive:
                # Mixed signals - negative takes priority (conservative approach)
                logger.info(f"LLM mixed signals: both positive and negative indicators found. "
                          f"Defaulting to False (conservative). Reason: {reason[:150]}")
                should_merge = False
            # else: no clear signals - trust should_merge as-is
        
        return {
            "should_merge": should_merge,
            "confidence": confidence,
            "reason": reason,
            "strategy": "llm_review",
        }

    except ImportError:
        logger.warning("Dify client not available for LLM review")
        return {
            "should_merge": False,
            "confidence": 0.0,
            "reason": "LLM not available (Dify client not installed)",
            "strategy": "llm_review_unavailable",
        }
    except Exception as e:
        logger.error(f"LLM review error: {e}")
        return {
            "should_merge": False,
            "confidence": 0.0,
            "reason": f"LLM review error: {str(e)[:100]}",
            "strategy": "llm_review_error",
        }


def review_pending_entities_with_llm(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    max_batches: int = 10,
    batch_size: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Review pending entities with LLM and auto-merge high-confidence matches.

    Args:
        nodes: List of entity nodes (may include review-flagged entities).
        relationships: List of relationships (for context matching).
        max_batches: Maximum number of LLM batches to process.
        batch_size: Number of entities per batch.

    Returns:
        (consolidated_nodes, id_remap, llm_stats)
    """
    llm_stats = {
        "total_reviewed": 0,
        "auto_merged": 0,
        "rejected": 0,
        "errors": 0,
        "batches_processed": 0,
    }

    # Find entities with review_status = "entity_match_review"
    review_entities = [
        n for n in nodes
        if isinstance(n, dict) and n.get("review_status") == "entity_match_review"
    ]

    if not review_entities:
        logger.info("No pending review entities found for LLM review")
        return nodes, {}, llm_stats

    logger.info(f"Found {len(review_entities)} entities pending LLM review")

    # Group by review_reason to find the "pair" entity they should be compared with
    # The review_reason contains the name of the potential match
    id_remap: dict[str, str] = {}
    reviewed_ids: set[str] = set()

    # Build name-to-entity map for quick lookup
    name_map: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict):
            norm_name = _normalize_text(_get_entity_name(n))
            if norm_name:
                name_map[norm_name] = n

    # Process in batches
    batches_processed = 0
    for batch_start in range(0, min(len(review_entities), max_batches * batch_size), batch_size):
        if batches_processed >= max_batches:
            logger.info(f"Reached max_batches limit ({max_batches}), stopping LLM review")
            break

        batch = review_entities[batch_start:batch_start + batch_size]
        batch_merged = 0

        for entity in batch:
            entity_id = _get_entity_id(entity)
            entity_name = _get_entity_name(entity)
            review_reason = entity.get("review_reason", "")

            # Extract the potential match name from review_reason
            # Format: "Possible match with '<name>' (strategy=xxx, score=xxx)"
            import re
            match = re.search(r"Possible match with '([^']+)'", review_reason)
            if not match:
                continue

            potential_match_name = match.group(1)

            # Find the potential match entity
            norm_match_name = _normalize_text(potential_match_name)
            potential_match = name_map.get(norm_match_name)

            if not potential_match:
                # Try partial match
                for norm_name, ent in name_map.items():
                    if norm_match_name in norm_name or norm_name in norm_match_name:
                        potential_match = ent
                        break

            if not potential_match:
                llm_stats["errors"] += 1
                continue

            # Skip if same entity
            match_id = _get_entity_id(potential_match)
            if entity_id == match_id or entity_id in reviewed_ids:
                continue

            # Call LLM for review
            llm_result = _llm_review_match(entity, potential_match)
            llm_stats["total_reviewed"] += 1

            if llm_result["should_merge"] and llm_result["confidence"] >= 0.6:
                # Auto-merge
                id_remap[entity_id] = match_id
                _merge_fields(potential_match, entity)
                reviewed_ids.add(entity_id)
                llm_stats["auto_merged"] += 1
                batch_merged += 1
                logger.info(f"[LLM-Review] Merged '{entity_name}' → '{_get_entity_name(potential_match)}' "
                           f"(confidence={llm_result['confidence']:.2f}, reason={llm_result['reason']})")
            else:
                # Reject merge, clear review status
                entity["review_status"] = "llm_review_rejected"
                entity["review_reason"] = f"LLM rejected: {llm_result['reason']}"
                llm_stats["rejected"] += 1

        batches_processed += 1
        llm_stats["batches_processed"] = batches_processed
        logger.info(f"[LLM-Review] Batch {batches_processed}: {len(batch)} entities reviewed, {batch_merged} merged")

    # Build consolidated nodes
    consolidated = [
        n for n in nodes
        if isinstance(n, dict) and _get_entity_id(n) not in reviewed_ids
    ]

    logger.info(f"[LLM-Review] Complete: {llm_stats['total_reviewed']} reviewed, "
                f"{llm_stats['auto_merged']} merged, {llm_stats['rejected']} rejected")

    return consolidated, id_remap, llm_stats


def export_review_checklist(
    nodes: list[dict[str, Any]],
    output_path: str,
    relationships: list[dict[str, Any]] | None = None,
) -> str:
    """Export pending review entities to a checklist file for manual review.

    Args:
        nodes: List of entity nodes.
        output_path: Path to write the checklist file.
        relationships: Optional relationships list for context.

    Returns:
        Path to the generated checklist file.
    """
    import json

    review_entities = [
        n for n in nodes
        if isinstance(n, dict) and n.get("review_status") == "entity_match_review"
    ]

    if not review_entities:
        logger.info("No pending review entities to export")
        return ""

    # Build name-to-entity map
    name_map: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict):
            norm_name = _normalize_text(_get_entity_name(n))
            if norm_name:
                name_map[norm_name] = n

    # Build checklist entries
    checklist = []
    for entity in review_entities:
        entity_id = _get_entity_id(entity)
        entity_name = _get_entity_name(entity)
        review_reason = entity.get("review_reason", "")

        # Find the potential match
        import re
        match = re.search(r"Possible match with '([^']+)'", review_reason)
        potential_match_name = match.group(1) if match else "Unknown"

        # Find the match entity details
        norm_match_name = _normalize_text(potential_match_name)
        match_entity = name_map.get(norm_match_name, {})

        # Get relationship context
        rel_context = []
        if relationships:
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                src = str(rel.get("source") or rel.get("from") or "")
                tgt = str(rel.get("target") or rel.get("to") or "")
                if src == entity_id or tgt == entity_id:
                    rel_type = rel.get("relation_type") or rel.get("type") or "unknown"
                    rel_context.append({
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                    })

        checklist.append({
            "entity_id": entity_id,
            "entity_name": entity_name,
            "entity_class": entity.get("meta_class", ""),
            "entity_layer": entity.get("layer", ""),
            "entity_description": entity.get("description", ""),
            "potential_match": {
                "name": potential_match_name,
                "id": _get_entity_id(match_entity) if match_entity else "",
                "class": match_entity.get("meta_class", "") if match_entity else "",
                "layer": match_entity.get("layer", "") if match_entity else "",
                "description": match_entity.get("description", "") if match_entity else "",
            },
            "review_reason": review_reason,
            "relationship_context": rel_context[:5],  # Limit to 5 relationships
            "action": "pending",  # Will be filled by manual review
            "reviewer": "",
            "notes": "",
        })

    # Write to file
    import time
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_pending": len(checklist),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checklist": checklist,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(checklist)} review items to {output_path}")
    return output_path


def apply_review_decisions(
    nodes: list[dict[str, Any]],
    decisions_path: str,
    relationships: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str], int]:
    """Apply manual review decisions from a checklist file.

    Args:
        nodes: List of entity nodes.
        decisions_path: Path to the completed checklist file (with actions filled in).
        relationships: Optional relationships list (for redirection).

    Returns:
        (consolidated_nodes, id_remap, applied_count)
    """
    import json

    if not os.path.exists(decisions_path):
        raise FileNotFoundError(f"Decisions file not found: {decisions_path}")

    with open(decisions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    decisions = data.get("checklist", [])
    id_remap: dict[str, str] = {}
    merged_ids: set[str] = set()
    applied_count = 0

    # Build entity lookup
    entity_map: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict):
            eid = _get_entity_id(n)
            if eid:
                entity_map[eid] = n

    for decision in decisions:
        entity_id = decision.get("entity_id", "")
        action = decision.get("action", "")
        match_id = decision.get("potential_match", {}).get("id", "")

        if action == "merge" and entity_id and match_id:
            loser = entity_map.get(entity_id)
            winner = entity_map.get(match_id)
            if loser and winner and entity_id != match_id:
                id_remap[entity_id] = match_id
                _merge_fields(winner, loser)
                merged_ids.add(entity_id)
                applied_count += 1
                logger.info(f"[Review] Applied merge: '{_get_entity_name(loser)}' → '{_get_entity_name(winner)}'")

    # Build consolidated nodes
    consolidated = [
        n for n in nodes
        if isinstance(n, dict) and _get_entity_id(n) not in merged_ids
    ]

    # Clear review status for non-merged entities
    for n in consolidated:
        if isinstance(n, dict) and n.get("review_status") == "entity_match_review":
            n["review_status"] = "manual_review_pending"

    logger.info(f"[Review] Applied {applied_count} merge decisions")
    return consolidated, id_remap, applied_count
