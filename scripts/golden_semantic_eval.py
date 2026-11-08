"""Golden semantic evaluation (E, 2026-08-15).

Semantic-regression gate: the existing regression runner validates thresholds
(counts / quality / related-to share) but NOT whether the extracted entities /
relations are *correct*. This module adds a golden-set semantics layer:

- A golden set declares the entities and relations that MUST be extracted from a
  sample (by id, or by a stable (name, meta_class) key + layer), plus relations
  that MUST NOT be emitted (rejected set).
- ``golden_metrics`` computes recall / precision / F1 against observed output,
  weighting semantic matches (key + layer + class) over raw name equality.

Pure and deterministic — no LLM, no I/O (all inputs are passed in), so it can be
unit-tested and wired into the regression runner without network cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", (name or "").lower()).strip()


def _tokens(name: str) -> List[str]:
    n = _norm(name)
    if not n:
        return []
    parts = re.split(r"\s+", n)
    # 中文单字也要切 token（支持「物料清单」→「物料」「清单」与「BOM 物料清单」匹配）
    expanded: List[str] = []
    for p in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", p):
            expanded.extend(list(p))
        else:
            expanded.append(p)
    return [t for t in expanded if t]


# --- P3：伞类 meta_class 放宽 ---
# 顶层 L1/L2 分类常见是 Data/System/Asset/Concept/Process/Organization，
# golden 里写 "System" 但实际 extract 可能写成 "Data"，这不应该阻挡语义召回。
_UMBRELLA_CLASS_BUCKETS: List[set] = [
    {"system", "asset", "concept", "data", "process", "organization", "service", "product", "object", "module"},
    {"person", "role", "actor", "user", "stakeholder", "team", "department"},
    {"metric", "kpi", "indicator", "measure", "index", "value", "ratio"},
    {"place", "location", "plant", "site", "warehouse", "factory", "facility", "region", "country"},
    {"document", "report", "file", "contract", "record", "artifact"},
]


def _class_bucket(klass: str) -> Optional[int]:
    k = _norm(klass)
    if not k:
        return None
    for idx, bucket in enumerate(_UMBRELLA_CLASS_BUCKETS):
        if k in bucket:
            return idx
    return None


def _classes_equivalent(gold_class: str, obs_class: str) -> bool:
    gc = _norm(gold_class)
    oc = _norm(obs_class)
    if not gc or not oc:
        return True  # 任一侧 class 未知就放行（宽松侧）
    if gc == oc:
        return True
    gb = _class_bucket(gc)
    ob = _class_bucket(oc)
    if gb is not None and ob is not None and gb == ob:
        return True
    return False


# --- P3：名称 token 级包含匹配 ---
def _name_match(gold_name: str, obs_name: str, min_overlap: float = 0.8) -> bool:
    """判断两个名称的 token 重叠度：gold 的所有 token 至少 80% 被 obs 包含即召回成功。

    场景：
      - gold="MBOM" ⊂ obs="plant mbom management" ✅
      - gold="BOM" ⊂ obs="Engineering BOM System" ✅
      - gold="主数据" （每个字命中）⊂ obs="主数据管理模块" ✅
      - gold="MASTER DATA" → [master,data] ⊂ obs="Product Master Data Management" ✅
    """
    gt = _tokens(gold_name)
    ot = _tokens(obs_name)
    if not gt or not ot:
        return False
    ot_set = set(ot)
    hits = sum(1 for tok in gt if tok in ot_set)
    # 短名（单 token）允许子串包含兜底（处理 BOM/MBOM/EBOM 这类纯缩写）
    if len(gt) == 1 and len(gt[0]) >= 2:
        single = gt[0]
        for tok in ot:
            if len(tok) >= 2 and (single in tok or tok in single):
                return True
    return hits / len(gt) >= min_overlap


def _entity_key(entity: Dict[str, Any]) -> Optional[tuple[str, str]]:
    """Stable match key for an entity: primary id if present, else (name, class).

    Returns None for entities that cannot be keyed (no id and no name).
    """
    if not isinstance(entity, dict):
        return None
    eid = str(entity.get("meta_id") or entity.get("entity_id") or entity.get("id") or "").strip()
    if eid:
        return (eid, "")
    name = str(entity.get("name") or entity.get("canonical_name") or entity.get("meta_name") or "").strip()
    klass = str(entity.get("meta_class") or entity.get("type") or entity.get("entity_type") or "").strip()
    if not name:
        return None
    return (name.lower(), klass.lower())


def _entity_layer(entity: Dict[str, Any]) -> str:
    return str(entity.get("layer") or "").strip().upper()


def _relation_key(rel: Dict[str, Any]) -> Optional[tuple[str, str, str]]:
    """Endpoint + type key for a relation (source_id, target_id, relation_type)."""
    if not isinstance(rel, dict):
        return None
    source = str(
        rel.get("source_id") or rel.get("source") or rel.get("from") or ""
    ).strip()
    target = str(
        rel.get("target_id") or rel.get("target") or rel.get("to") or ""
    ).strip()
    rtype = str(
        rel.get("relation_type") or rel.get("relationship_type") or rel.get("type") or ""
    ).strip().upper()
    if not source or not target or not rtype:
        return None
    return (source, target, rtype)


@dataclass
class GoldenEntity:
    name: str
    meta_class: str = ""
    layer: str = ""
    aliases: List[str] = field(default_factory=list)

    def match_keys(self) -> List[tuple[str, str]]:
        keys = [(self.name.lower(), self.meta_class.lower())]
        for alias in self.aliases:
            if alias:
                keys.append((alias.lower(), self.meta_class.lower()))
        return keys


@dataclass
class GoldenRelation:
    """A golden relation by endpoint names + relation types (name-based)."""
    source_name: str
    target_name: str
    relation_types: List[str]
    min_confidence: float = 0.0


@dataclass
class GoldenSample:
    sample_id: str
    entities: List[GoldenEntity] = field(default_factory=list)
    relations: List[GoldenRelation] = field(default_factory=list)
    rejected_entities: List[GoldenEntity] = field(default_factory=list)
    reject_relation_types: List[str] = field(default_factory=list)


def golden_metrics(
    sample: GoldenSample,
    observed_entities: List[Dict[str, Any]],
    observed_relations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute semantic recall/precision/F1 for a golden sample.

    Entity matching is by stable key (id, or name+class), and requires the
    observed layer to match the golden layer when the golden declares one.
    Absent distractors (no rejected set provided) precision defaults to 1.0.
    """
    # ---- entity recall ----
    # 按 (name, class) 与 id 建立索引，用于 layer 门禁下回查具体 observed 实体
    observed_by_name_class: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    observed_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for ent in observed_entities:
        key = _entity_key(ent)
        if key:
            if key[1]:
                observed_by_name_class.setdefault(key, []).append(ent)
            else:
                observed_by_id.setdefault(key[0], []).append(ent)
            # Name-based key too: golden sets declare entities by name+class even
            # when the extractor keys them by id. Match either id- or name-key.
            name = str(ent.get("name") or ent.get("canonical_name") or ent.get("meta_name") or "").strip()
            klass = str(ent.get("meta_class") or ent.get("type") or ent.get("entity_type") or "").strip()
            if name:
                observed_by_name_class.setdefault((name.lower(), klass.lower()), []).append(ent)

    # 构建 observed 实体索引（name/class/layer），用于后续伞类+token 级模糊匹配遍历
    observed_entity_flat: List[Dict[str, Any]] = []
    for ent in observed_entities:
        name = str(ent.get("name") or ent.get("canonical_name") or ent.get("meta_name") or "").strip()
        klass = str(ent.get("meta_class") or ent.get("type") or ent.get("entity_type") or "").strip()
        layer = _entity_layer(ent)
        if name:
            observed_entity_flat.append({"name": name, "class": klass, "layer": layer})

    def _layer_gate_passes(gold_layer: str, obs: Dict[str, Any]) -> bool:
        """golden 声明了 layer 时，observed 必须同 layer 才算匹配。"""
        gl = gold_layer.strip().upper()
        if not gl:
            return True
        ol = _entity_layer(obs)
        return bool(ol) and ol == gl

    def _candidates_for(gold: GoldenEntity) -> List[Dict[str, Any]]:
        cands: List[Dict[str, Any]] = []
        seen: set = set()
        for key in gold.match_keys():
            for ent in observed_by_name_class.get(key, []):
                if id(ent) not in seen:
                    seen.add(id(ent))
                    cands.append(ent)
            for ent in observed_by_id.get(key[0], []):
                if id(ent) not in seen:
                    seen.add(id(ent))
                    cands.append(ent)
        alias_names = [gold.name] + list(gold.aliases or [])
        for gname in alias_names:
            for obs in observed_entity_flat:
                if not obs.get("name"):
                    continue
                if not _classes_equivalent(gold.meta_class, obs["class"]):
                    continue
                if _name_match(gname, obs["name"]) and id(obs) not in seen:
                    seen.add(id(obs))
                    cands.append(obs)
        return cands

    matched_required = 0
    unmatched_required: List[Dict[str, Any]] = []
    for gold in sample.entities:
        # 1) 精确 key 匹配（向后兼容：id 或 name+class 完全相等）
        # 2) P3：伞类放宽 + name token 包含匹配
        # 3) layer 门禁：golden 声明了 layer 时，observed 的 layer 必须一致，
        #    layer 是细粒度分类的一部分，偏差不应计入召回。
        candidates = _candidates_for(gold)
        hit = any(_layer_gate_passes(gold.layer, c) for c in candidates)
        if hit:
            matched_required += 1
        else:
            unmatched_required.append({"name": gold.name, "meta_class": gold.meta_class, "layer": gold.layer})

    # ---- rejected entities ----
    false_positives = []
    rejected_keys = set()
    for gold in sample.rejected_entities:
        rejected_keys.update(gold.match_keys())
        rejected_keys.update((gold.name.lower(), gold.meta_class.lower()))
    for ent in observed_entities:
        key = _entity_key(ent)
        if key and key in rejected_keys:
            false_positives.append({
                "entity": str(ent.get("name") or ent.get("meta_id") or ""),
                "reason": "rejected entity emitted",
            })

    # ---- relation recall (name-based) ----
    # Phase B 双层评估口径 fix (2026-08-29): the endpoint ID space must be
    # normalized to a SINGLE case (lower) on BOTH sides.  Previously
    # ``resolved_id`` kept the raw-case eid while observed relation endpoints
    # fell through to ``name.lower()``, so tuple membership checks
    # ("L1-001" vs "l1-001") always missed and relation_recall collapsed to 0.
    name_to_ids: Dict[str, List[str]] = {}
    known_ids: set = set()
    for ent in observed_entities:
        name = str(ent.get("name") or ent.get("canonical_name") or ent.get("meta_name") or "").strip()
        eid = str(ent.get("meta_id") or ent.get("entity_id") or ent.get("id") or "").strip()
        if name:
            resolved_id = (eid or name).lower()
            known_ids.add(resolved_id)
            name_to_ids.setdefault(name.lower(), []).append(resolved_id)
            for alias in ent.get("aliases") or []:
                alias_name = str(alias or "").strip().lower()
                if alias_name:
                    name_to_ids.setdefault(alias_name, []).append(resolved_id)

    def resolve(name: str) -> List[str]:
        exact = name_to_ids.get(name.lower())
        if exact:
            return exact
        # Observed relation endpoints may carry the entity id directly
        # (e.g. "L1-001"); pass them through in the same lowercase space.
        if name.lower() in known_ids:
            return [name.lower()]
        # Golden relations use human names while some observed relations use
        # normalized variants. Resolve only when one observed entity is an
        # unambiguous token-level name match; do not silently merge endpoints.
        candidates: List[str] = []
        for observed_name, ids in name_to_ids.items():
            if _name_match(name, observed_name) and len(ids) == 1:
                candidates.extend(ids)
        return sorted(set(candidates)) or [name.lower()]

    rel_by_ids: set = set()
    for rel in observed_relations:
        key = _relation_key(rel)
        if not key:
            continue
        rsrc, rtgt, rtype = key
        # Resolve observed endpoints through the same name→id space used by
        # the golden side, otherwise name-based relations (e.g. safety-net
        # EBOM→BOM) can never match id-based golden endpoints.
        for s in resolve(rsrc):
            for t in resolve(rtgt):
                rel_by_ids.add((s, t, rtype))

    matched_relations = 0
    unmatched_relations = []
    for gold in sample.relations:
        src_ids = resolve(gold.source_name)
        tgt_ids = resolve(gold.target_name)
        accepted_types = {t.upper() for t in gold.relation_types}
        found = False
        for s in src_ids:
            for t in tgt_ids:
                if any((s, t, rt) in rel_by_ids for rt in accepted_types) or any((t, s, rt) in rel_by_ids for rt in accepted_types):
                    found = True
                    break
            if found:
                break
        if found:
            matched_relations += 1
        else:
            unmatched_relations.append({"source": gold.source_name, "target": gold.target_name, "types": gold.relation_types})

    # ---- rejected relation types ----
    rejected_observations = []
    if sample.reject_relation_types:
        rejected_set = {r.upper() for r in sample.reject_relation_types}
        for rel in observed_relations:
            key = _relation_key(rel)
            if key and key[2] in rejected_set:
                rejected_observations.append({"relation_type": key[2], "source": key[0], "target": key[1]})

    # ---- aggregate ----
    total_required = len(sample.entities)
    entity_recall = round(matched_required / total_required, 4) if total_required else 1.0
    entity_precision = round(
        max(0.0, 1.0 - len(false_positives) / max(len(observed_entities), 1)),
        4,
    ) if sample.rejected_entities else 1.0

    total_relations = len(sample.relations)
    relation_recall = round(matched_relations / total_relations, 4) if total_relations else 1.0

    precision = entity_precision
    recall = (entity_recall + relation_recall) / 2.0 if total_relations else entity_recall
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    return {
        "entity_recall": entity_recall,
        "entity_precision": entity_precision,
        "relation_recall": relation_recall,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": f1,
        "matched_entities": matched_required,
        "total_entities": total_required,
        "unmatched_entities": unmatched_required,
        "matched_relations": matched_relations,
        "total_relations": total_relations,
        "unmatched_relations": unmatched_relations,
        "false_positives": false_positives,
        "rejected_relation_observations": rejected_observations,
        "ok": (
            entity_recall >= 0.8
            and relation_recall >= 0.6
            and not false_positives
            and not rejected_observations
        ),
    }


def cq_provenance_stats(relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """T2.3 CQ provenance registry (ENTITY_CLASSIFICATION_ROADMAP).

    Count "dead relations" — edges with no CQ 出处 (empty/absent
    ``source_cq``). Per the CQ-driven methodology, every relation should be
    attributable to at least one competency question; edges without one have
    no registered business justification.

    Pure function over packet-style relation dicts (as produced by
    ``extraction_packet_adapter.adapt_extraction_json_to_packet``).
    """
    total = len(relations)
    dead_by_type: Dict[str, int] = {}
    with_cq = 0
    for rel in relations:
        cq = str(rel.get("source_cq") or "").strip() if isinstance(rel, dict) else ""
        if cq:
            with_cq += 1
        else:
            rtype = str(rel.get("relation_type") or "?") if isinstance(rel, dict) else "?"
            dead_by_type[rtype] = dead_by_type.get(rtype, 0) + 1
    dead = total - with_cq
    return {
        "total_relations": total,
        "with_source_cq": with_cq,
        "dead_relations": dead,
        "dead_ratio": round(dead / total, 4) if total else 0.0,
        "dead_by_type": dead_by_type,
    }


def load_golden_set(data: Dict[str, Any]) -> GoldenSample:
    """Load a golden semantic set from manifest JSON block."""
    entities = []
    for ent in (data.get("entities") or []):
        entities.append(GoldenEntity(
            name=str(ent.get("name") or ""),
            meta_class=str(ent.get("meta_class") or ent.get("type") or ""),
            layer=str(ent.get("layer") or ""),
            aliases=list(ent.get("aliases") or []),
        ))
    relations = []
    for rel in (data.get("relations") or []):
        relations.append(GoldenRelation(
            source_name=str(rel.get("source") or ""),
            target_name=str(rel.get("target") or ""),
            relation_types=[str(t) for t in (rel.get("types") or [])],
            min_confidence=float(rel.get("min_confidence", 0.0) or 0.0),
        ))
    rejected_entities = []
    for ent in (data.get("rejected_entities") or []):
        rejected_entities.append(GoldenEntity(
            name=str(ent.get("name") or ""),
            meta_class=str(ent.get("meta_class") or ent.get("type") or ""),
            layer=str(ent.get("layer") or ""),
            aliases=list(ent.get("aliases") or []),
        ))
    return GoldenSample(
        sample_id=str(data.get("sample_id") or ""),
        entities=entities,
        relations=relations,
        rejected_entities=rejected_entities,
        reject_relation_types=[str(r).upper() for r in (data.get("reject_relation_types") or [])],
    )