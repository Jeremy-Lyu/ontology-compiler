"""P2-1 (1): per-query channel router (lexical / vector / graph / structured).

Pure, deterministic rules over the ALREADY-parsed query surface
(``ParsedQuery`` + ``QueryClassification`` from ``domain.query_core``) —
no new LLM calls, no model dependency. The router answers "which channels
deserve budget for THIS query" so downstream fusion only fuses channels
that fired.

Channel semantics:
- lexical (BM25): always on — exact term match is the recall floor.
- vector (semantic): on unless the query is a pure entity/attribute lookup
  (exact-name queries gain nothing from embedding fuzz).
- graph (structure): on for relation/comparison/cause-effect intent or
  multi-entity queries (needs entity-relation traversal, not just chunks).
- structured (index): on for entity/attribute/list/validation queries
  (metadata-filtered lookup beats free-text ranking).

Output is a frozenset of channel names + the reason map, so callers can
log routing decisions and the eval harness can assert stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet

LEXICAL = "lexical"
VECTOR = "vector"
GRAPH = "graph"
STRUCTURED = "structured"

ALL_CHANNELS: FrozenSet[str] = frozenset({LEXICAL, VECTOR, GRAPH, STRUCTURED})

_RELATION_INTENTS = {"relationship", "comparison", "cause_effect"}
_STRUCTURED_INTENTS = {"definition", "validation", "list"}
_LOOKUP_TYPES = {"entity_lookup", "attribute_query"}


@dataclass(frozen=True)
class ChannelDecision:
    channels: FrozenSet[str]
    reasons: Dict[str, str] = field(default_factory=dict)

    def __contains__(self, channel: str) -> bool:
        return channel in self.channels


def route_channels(
    *,
    query_type: str = "",
    intent: str = "",
    is_multi_entity: bool = False,
    entity_count: int = 0,
) -> ChannelDecision:
    """Route a parsed query to retrieval channels (pure function)."""
    channels: set[str] = {LEXICAL}
    reasons: Dict[str, str] = {LEXICAL: "recall floor: exact term match always on"}

    if query_type in _LOOKUP_TYPES and not is_multi_entity:
        reasons[VECTOR] = "off: pure exact-name lookup gains nothing from embedding fuzz"
    else:
        channels.add(VECTOR)
        reasons[VECTOR] = "on: semantic match for non-lookup query"

    if intent in _RELATION_INTENTS or is_multi_entity or entity_count >= 2:
        channels.add(GRAPH)
        reasons[GRAPH] = f"on: intent={intent} multi={is_multi_entity} needs traversal"
    else:
        reasons[GRAPH] = "off: single-entity non-relational query"

    if intent in _STRUCTURED_INTENTS or query_type in _LOOKUP_TYPES:
        channels.add(STRUCTURED)
        reasons[STRUCTURED] = f"on: intent={intent} type={query_type} suits metadata lookup"
    else:
        reasons[STRUCTURED] = "off: free-text ranking suffices"

    return ChannelDecision(channels=frozenset(channels), reasons=reasons)


def route_parsed(parsed_query=None, classification=None) -> ChannelDecision:
    """Convenience: route from query_core objects (duck-typed, no import)."""
    qtype = str(getattr(classification, "query_type", "") or "")
    if hasattr(qtype, "value"):
        qtype = qtype.value
    intent = str(getattr(classification, "intent", "") or "")
    if hasattr(intent, "value"):
        intent = intent.value
    # query_type on ParsedQuery is absent; QueryClassification carries it.
    return route_channels(
        query_type=qtype,
        intent=intent,
        is_multi_entity=bool(getattr(parsed_query, "is_multi_entity", False)),
        entity_count=len(getattr(parsed_query, "entities", []) or []),
    )
