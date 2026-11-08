"""Meaning-first surface over the governed OntologyPacket.

Inspired by NVIDIA Auto Ontology's MCP allow-list: agents should reach
business meaning (entities, relations, answerability) rather than dump raw
master JSON or filesystem paths.

Public operations
-----------------
* ``search_entities`` — glossary-style lookup by name / alias / type / id
* ``get_entity`` — one entity plus incident relations
* ``check_answerable`` — cheap preflight before RAG / LLM answer
* ``load_governed_packet`` — packet.json from the current governed dir

This module does not execute SQL and does not expose storage layout.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.application.rag.governed_packets import default_governed_rag_dir

__all__ = [
    "EntityHit",
    "AnswerabilityReport",
    "flatten_entities",
    "load_governed_packet",
    "search_entities",
    "get_entity",
    "check_answerable",
    "ask_question",
    "relation_type_hints",
]

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]{2,}", re.UNICODE)

# Question cue → expected relation_type(s). Keep small and explicit.
_RELATION_CUES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"派生|衍生|derived|derive", re.I), ("DERIVED_FROM",)),
    (re.compile(r"包含|组成|consists|contains|contain", re.I), ("CONTAINS", "PART_OF")),
    (re.compile(r"属于|归属|belongs", re.I), ("BELONGS_TO", "PART_OF")),
    (re.compile(r"依赖|depends", re.I), ("DEPENDS_ON",)),
    (re.compile(r"实现|implements", re.I), ("IMPLEMENTS",)),
    (re.compile(r"使用|uses|using", re.I), ("USES",)),
    (re.compile(r"发送|sent", re.I), ("SENT_TO",)),
)

# Common manufacturing concepts agents ask about; match loosely against names.
_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "ebom": ("ebom", "工程bom", "engineering bom", "工程物料清单"),
    "mbom": ("mbom", "制造bom", "manufacturing bom", "制造物料清单"),
    "bom": ("bom", "物料清单", "bill of materials"),
    "ecn": ("ecn", "工程变更", "engineering change"),
    "零件": ("零件", "part", "component", "物料"),
    "工艺": ("工艺", "process", "routing"),
    "工厂": ("工厂", "plant", "factory"),
}


@dataclass(frozen=True)
class EntityHit:
    id: str
    name: str
    layer: str
    type: str
    aliases: tuple[str, ...] = ()
    score: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerabilityReport:
    answerable: bool
    confidence: float
    question: str
    matched_entities: List[EntityHit] = field(default_factory=list)
    matched_relation_types: List[str] = field(default_factory=list)
    missing_concepts: List[str] = field(default_factory=list)
    missing_relation_types: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    packet_id: str = ""
    entity_count: int = 0
    relation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["matched_entities"] = [e if isinstance(e, dict) else e.to_dict() for e in self.matched_entities]
        return payload


def load_governed_packet(packet_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``packet.json`` from a governed directory (latest if omitted)."""
    root = Path(packet_dir) if packet_dir is not None else default_governed_rag_dir()
    path = root / "packet.json"
    if not path.exists():
        raise FileNotFoundError(f"governed packet.json missing at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_entities(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    buckets = packet.get("entities") or {}
    if isinstance(buckets, dict):
        out: List[Dict[str, Any]] = []
        for layer, items in buckets.items():
            if not isinstance(items, list):
                continue
            for ent in items:
                if not isinstance(ent, dict):
                    continue
                row = dict(ent)
                row.setdefault("layer", layer)
                out.append(row)
        return out
    if isinstance(buckets, list):
        return [e for e in buckets if isinstance(e, dict)]
    return []


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def _entity_strings(ent: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for key in ("id", "name", "type", "code", "meta_id", "meta_name", "meta_class"):
        val = ent.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    aliases = ent.get("aliases") or []
    if isinstance(aliases, list):
        parts.extend(str(a) for a in aliases if a)
    tags = ent.get("tags") or []
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags if t)
    return parts


def search_entities(
    query: str,
    *,
    packet: Optional[Dict[str, Any]] = None,
    packet_dir: Optional[Path] = None,
    limit: int = 10,
    layers: Optional[Sequence[str]] = None,
) -> List[EntityHit]:
    """Search entities by substring / token overlap on name, alias, type, id."""
    pkt = packet if packet is not None else load_governed_packet(packet_dir)
    q = _norm(query)
    if not q:
        return []
    layer_filter = {str(x) for x in layers} if layers else None
    hits: List[EntityHit] = []
    for ent in flatten_entities(pkt):
        layer = str(ent.get("layer") or "")
        if layer_filter is not None and layer not in layer_filter:
            continue
        strings = _entity_strings(ent)
        norms = [_norm(s) for s in strings]
        score = 0.0
        if any(q == n for n in norms):
            score = 1.0
        elif any(q in n or n in q for n in norms if n):
            score = 0.85
        else:
            tokens = {t.lower() for t in _TOKEN_RE.findall(query)}
            blob = " ".join(norms)
            overlap = sum(1 for t in tokens if t and t in blob)
            if overlap:
                score = min(0.7, 0.25 * overlap)
        if score <= 0:
            continue
        aliases = tuple(str(a) for a in (ent.get("aliases") or []) if a)
        hits.append(
            EntityHit(
                id=str(ent.get("id") or ent.get("meta_id") or ""),
                name=str(ent.get("name") or ent.get("meta_name") or ""),
                layer=layer,
                type=str(ent.get("type") or ent.get("meta_class") or ""),
                aliases=aliases,
                score=score,
                description=str(ent.get("description") or "")[:240],
            )
        )
    hits.sort(key=lambda h: (-h.score, h.layer, h.name))
    return hits[: max(1, limit)]


def get_entity(
    entity_id: str,
    *,
    packet: Optional[Dict[str, Any]] = None,
    packet_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return one entity plus incident relations (meaning view, not storage)."""
    pkt = packet if packet is not None else load_governed_packet(packet_dir)
    target = None
    for ent in flatten_entities(pkt):
        eid = str(ent.get("id") or ent.get("meta_id") or "")
        if eid == entity_id:
            target = ent
            break
    if target is None:
        return None
    rels = [r for r in (pkt.get("relations") or []) if isinstance(r, dict)]
    incident = [
        r
        for r in rels
        if str(r.get("source_id") or "") == entity_id or str(r.get("target_id") or "") == entity_id
    ]
    return {
        "entity": {
            "id": target.get("id") or target.get("meta_id"),
            "name": target.get("name") or target.get("meta_name"),
            "layer": target.get("layer"),
            "type": target.get("type") or target.get("meta_class"),
            "code": target.get("code"),
            "description": target.get("description"),
            "aliases": target.get("aliases") or [],
            "status": target.get("status"),
            "evidence_ref": target.get("evidence_ref"),
            "attributes": target.get("attributes") or {},
        },
        "relations": incident,
        "relation_count": len(incident),
    }


def relation_type_hints(question: str) -> List[str]:
    found: List[str] = []
    for pattern, types in _RELATION_CUES:
        if pattern.search(question or ""):
            for t in types:
                if t not in found:
                    found.append(t)
    return found


def _concept_mentions(question: str) -> List[str]:
    qn = _norm(question)
    mentioned: List[str] = []
    for label, aliases in _CONCEPT_ALIASES.items():
        if any(_norm(a) and _norm(a) in qn for a in (label, *aliases)):
            mentioned.append(label)
    return mentioned


def check_answerable(
    question: str,
    *,
    packet: Optional[Dict[str, Any]] = None,
    packet_dir: Optional[Path] = None,
    min_entity_hits: int = 1,
) -> AnswerabilityReport:
    """Grade whether the governed packet can plausibly ground ``question``.

    Fail closed on empty packet or zero concept coverage when the question
    clearly names known manufacturing concepts that are absent.
    """
    pkt = packet if packet is not None else load_governed_packet(packet_dir)
    entities = flatten_entities(pkt)
    relations = [r for r in (pkt.get("relations") or []) if isinstance(r, dict)]
    present_rel_types = {
        str(r.get("relation_type") or r.get("type") or "").upper()
        for r in relations
        if r.get("relation_type") or r.get("type")
    }

    packet_id = ""
    try:
        from src.application.rag.governed_packets import read_latest_pointer

        pointer = read_latest_pointer() or {}
        packet_id = str(pointer.get("packet_id") or "")
    except Exception:
        packet_id = ""

    gaps: List[str] = []
    missing_concepts: List[str] = []
    missing_rels: List[str] = []

    if not entities:
        gaps.append("governed packet has zero entities")
        return AnswerabilityReport(
            answerable=False,
            confidence=0.0,
            question=question,
            gaps=gaps,
            packet_id=packet_id,
            entity_count=0,
            relation_count=len(relations),
        )

    hits = search_entities(question, packet=pkt, limit=8)
    # Concept coverage: EBOM/MBOM etc. must appear somewhere if named.
    for concept in _concept_mentions(question):
        aliases = _CONCEPT_ALIASES.get(concept, (concept,))
        covered = False
        for ent in entities:
            blob = _norm(" ".join(_entity_strings(ent)))
            if any(_norm(a) in blob for a in (concept, *aliases) if a):
                covered = True
                break
        if not covered:
            missing_concepts.append(concept)
            gaps.append(f"concept '{concept}' not found in governed entities")

    expected_rels = relation_type_hints(question)
    matched_rels = [t for t in expected_rels if t in present_rel_types]
    for t in expected_rels:
        if t not in present_rel_types:
            missing_rels.append(t)
            gaps.append(f"relation type '{t}' absent from governed packet")

    entity_ok = len(hits) >= min_entity_hits or (
        not _concept_mentions(question) and len(hits) == 0 and not expected_rels
    )
    # If the question names concepts or relations, require those to resolve.
    structural_ok = not missing_concepts and not missing_rels
    # Pure open questions with some entity hits are answerable; empty hits + cues are not.
    if _concept_mentions(question) or expected_rels:
        answerable = structural_ok and (len(hits) >= 1 or not missing_concepts)
        # If concepts resolved via alias scan but search_entities missed, still OK.
        if structural_ok and missing_concepts == [] and (_concept_mentions(question) or matched_rels):
            answerable = True
    else:
        answerable = len(hits) >= min_entity_hits
        if not answerable:
            gaps.append("no entity name/alias matched the question tokens")

    confidence = 0.0
    if answerable:
        confidence = 0.55
        if hits:
            confidence += 0.2 * min(1.0, hits[0].score)
        if matched_rels:
            confidence += 0.15
        if _concept_mentions(question) and not missing_concepts:
            confidence += 0.1
        confidence = min(0.99, confidence)
    else:
        confidence = 0.15 if hits else 0.05

    # entity_ok unused beyond clarity — keep lint clean
    _ = entity_ok

    return AnswerabilityReport(
        answerable=answerable,
        confidence=round(confidence, 3),
        question=question,
        matched_entities=hits,
        matched_relation_types=matched_rels,
        missing_concepts=missing_concepts,
        missing_relation_types=missing_rels,
        gaps=gaps,
        packet_id=packet_id,
        entity_count=len(entities),
        relation_count=len(relations),
    )


def ask_question(
    question: str,
    *,
    packet: Optional[Dict[str, Any]] = None,
    packet_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Answer only after answerability gate (fail-closed).

    Returns a grounded evidence sketch from matched entities / relation types.
    Does **not** call an LLM — agents that need generative RAG should use
    ``scripts/rag_cli.py query`` after this gate passes.
    """
    report = check_answerable(question, packet=packet, packet_dir=packet_dir)
    payload: Dict[str, Any] = {
        "question": question,
        "answerable": report.answerable,
        "confidence": report.confidence,
        "gaps": list(report.gaps),
        "packet_id": report.packet_id,
        "refused": False,
        "answer": None,
        "evidence": {
            "entities": [e.to_dict() for e in report.matched_entities[:5]],
            "relation_types": list(report.matched_relation_types),
        },
    }
    if not report.answerable and not force:
        payload["refused"] = True
        payload["answer"] = (
            "语义层覆盖不足，已拒绝作答。缺口: "
            + ("; ".join(report.gaps) if report.gaps else "unknown")
        )
        return payload

    lines: List[str] = []
    if report.matched_entities:
        names = ", ".join(f"{e.name}({e.layer})" for e in report.matched_entities[:5])
        lines.append(f"相关实体: {names}")
    if report.matched_relation_types:
        lines.append("相关关系类型: " + ", ".join(report.matched_relation_types))
    if report.missing_concepts:
        lines.append("仍缺概念: " + ", ".join(report.missing_concepts))
    if not lines:
        lines.append("packet 可答，但未匹配到具体实体名；请用 search_entities 继续检索。")
    payload["answer"] = "\n".join(lines)
    return payload
