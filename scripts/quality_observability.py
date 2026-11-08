"""Quality observability for regression samples (P1-5, Gate D).

Pure functions over a regression sample payload (stage_results + merged
master file): entity drift (extract→master), evidence coverage of the merged
master, and dead-relation (missing source_cq) stats. Warn-level only — never
affects pass/fail gates.

- ``quality_observability``: per-sample block attached by the regression
  runner next to ``diagnostics``.
- Batch aggregation reuses the layered-golden mean pattern; the runner merges
  per-sample blocks into ``quality_observability_summary``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _rel_type(rel: Dict[str, Any]) -> str:
    return str(rel.get("relation_type") or rel.get("type") or "?").upper()


def _rel_endpoints(rel: Dict[str, Any]) -> tuple[str, str]:
    src = str(rel.get("source") or rel.get("from") or rel.get("source_id") or "")
    tgt = str(rel.get("target") or rel.get("to") or rel.get("target_id") or "")
    return src, tgt


def _norm_name(name: Any) -> str:
    return str(name or "").strip().lower()


def _entity_names(entity: Dict[str, Any]) -> set[str]:
    names = {_norm_name(entity.get("name") or entity.get("meta_name"))}
    aliases = entity.get("aliases")
    if isinstance(aliases, list):
        names |= {_norm_name(a) for a in aliases}
    names.discard("")
    return names


def _is_ghost(entity: Dict[str, Any]) -> bool:
    desc = str(entity.get("description") or "")
    return "auto created ghost node" in desc.lower()


def entity_drift_extract_to_master(
    extract_entities: List[Dict[str, Any]],
    master_entities: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Name(+alias)-based drift between extract output and the merged master.

    Matches on normalized name AND aliases (merge may rename / fold instances
    into a canonical node, keeping the old name as alias — id-based diff
    would over-report churn even harder). Reports added (in master but not
    extract — typically baseline-inherited), missing (extracted but lost in
    merge — the retention-relevant signal), and matched counts.

    口径注意（2026-09-09 实证）：
    1. missing 通常是实例层归并 churn（global optimizer 的 cross-layer /
       same-layer / MSER 合并；抽取 ghost 节点"auto created ghost node"
       本就该被合并掉），不等于语义丢失——语义丢失以 layered_golden 的
       merge recall / retention 为准。
    2. ghost 节点单列 ``ghost_missing`` / ``ghost_missing_names``（merge 清洗
       ghost 属于治理成功而非回归；golden 从不含 ghost，双口径互不干扰）。
    """
    ghost_names: set[str] = set()
    extract_names: set[str] = set()
    for e in extract_entities:
        if isinstance(e, dict):
            ns = _entity_names(e)
            extract_names |= ns
            if _is_ghost(e):
                ghost_names |= ns
    master_names: set[str] = set()
    for e in master_entities:
        if isinstance(e, dict):
            master_names |= _entity_names(e)
    missing = sorted(extract_names - master_names)
    added = sorted(master_names - extract_names)
    matched = len(extract_names & master_names)
    total = max(len(extract_names), 1)
    ghost_missing = sorted(n for n in missing if n in ghost_names)
    real_missing = [n for n in missing if n not in ghost_names]
    return {
        "extract_entities": len(extract_names),
        "master_entities": len(master_names),
        "matched": matched,
        "missing_in_master": missing,
        "added_in_master": added,
        "missing_count": len(missing),
        "missing_ratio": round(len(missing) / total, 4),
        "ghost_missing_count": len(ghost_missing),
        "ghost_missing_names": ghost_missing,
        "real_missing_count": len(real_missing),
        "real_missing_names": real_missing,
        "real_missing_ratio": round(len(real_missing) / total, 4),
    }


def evidence_coverage_master(master_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evidence coverage of the merged master (nodes carrying evidence)."""
    total = len(master_entities)
    with_ev = sum(1 for e in master_entities if isinstance(e, dict) and e.get("evidence"))
    return {
        "total": total,
        "with_evidence": with_ev,
        "coverage": round(with_ev / total, 4) if total else 0.0,
    }


def dead_relation_stats(master_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dead-relation stats: edges with no traceable origin (no provenance AND no evidence).

    口径说明（2026-09-09 P1-5 定死）：生产链路中没有任何模块真正产出
    ``source_cq``（仅 extraction_packet_adapter 透传上游已有值、RAG bridge
    透传、master_ontology_adapter 注释明示 master 无 source_cq；唯一写
    source_cq 的是测试/e2e 与 golden_semantic_eval 的评估侧）。因此按
    "无 source_cq" 统计 dead 会在 merged master 上恒 ~100%，无信号价值。
    改为"来路不明"口径：既无 provenance（llm/skill/rule/inferred…）又无
    evidence/evidence_ref 的边才算 dead。``source_cq`` 覆盖度单独统计为
    ``with_source_cq`` / ``source_cq_coverage``（CQ 接入后自然有数），不进
    ``dead_ratio``。
    """
    total = len(master_relations)
    dead_by_type: Counter = Counter()
    with_cq = 0
    traceable = 0
    for rel in master_relations:
        if not isinstance(rel, dict):
            continue
        cq = str(rel.get("source_cq") or "").strip()
        if cq:
            with_cq += 1
        prov = str(rel.get("provenance") or "").strip()
        ev = rel.get("evidence") or rel.get("evidence_ref")
        has_ev = bool(ev) if not isinstance(ev, str) else bool(ev.strip())
        if prov or has_ev:
            traceable += 1
        else:
            dead_by_type[_rel_type(rel)] += 1
    dead = total - traceable
    return {
        "total_relations": total,
        "with_source_cq": with_cq,
        "source_cq_coverage": round(with_cq / total, 4) if total else 0.0,
        "traceable_relations": traceable,
        "dead_relations": dead,
        "dead_ratio": round(dead / total, 4) if total else 0.0,
        "dead_by_type": dict(dead_by_type),
    }


def related_to_share_master(master_relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """RELATED_TO fallback share in the merged master."""
    total = len(master_relations)
    rt = sum(1 for r in master_relations if isinstance(r, dict) and _rel_type(r) == "RELATED_TO")
    return {
        "total": total,
        "related_to": rt,
        "share": round(rt / total, 4) if total else 0.0,
    }


def build_quality_observability(
    extract_entities: List[Dict[str, Any]],
    extract_relations: List[Dict[str, Any]],
    master_entities: List[Dict[str, Any]],
    master_relations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the per-sample quality_observability block (warn-only)."""
    _ = extract_relations  # reserved for future extract↔master relation drift
    return {
        "entity_drift": entity_drift_extract_to_master(extract_entities, master_entities),
        "evidence": evidence_coverage_master(master_entities),
        "dead_relations": dead_relation_stats(master_relations),
        "related_to": related_to_share_master(master_relations),
        "口径": {
            "entity_drift": "extract→master 按归一化 name 匹配（master remap meta_id，故不用 id）；missing_in_master=抽取后丢失（retention 相关信号）",
            "evidence": "merged master 节点带 evidence 比例",
            "dead_relations": "merged master 边来路不明比例（无 provenance 又无 evidence；2026-09-09 定死——source_cq 无生产者，恒~100% 无信号，单列 source_cq_coverage 追踪 CQ 接入）",
            "related_to": "merged master 中 RELATED_TO 兜底占比",
        },
    }


def aggregate_quality_observability(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch-level mean aggregation over per-sample observability blocks."""
    def means(key: str, sub: str) -> Dict[str, Any]:
        vals = [b.get(key, {}).get(sub) for b in blocks if isinstance(b.get(key, {}).get(sub), (int, float))]
        return {"mean": round(sum(vals) / len(vals), 4), "n": len(vals)} if vals else {"mean": None, "n": 0}

    missing_names: Counter = Counter()
    real_missing_names: Counter = Counter()
    dead_types: Counter = Counter()
    for b in blocks:
        for n in (b.get("entity_drift", {}).get("missing_in_master") or []):
            missing_names[str(n)] += 1
        for n in (b.get("entity_drift", {}).get("real_missing_names") or []):
            real_missing_names[str(n)] += 1
        for t, c in ((b.get("dead_relations", {}).get("dead_by_type")) or {}).items():
            dead_types[str(t)] += int(c or 0)
    return {
        "metrics": {
            "missing_ratio": means("entity_drift", "missing_ratio"),
            "real_missing_ratio": means("entity_drift", "real_missing_ratio"),
            "ghost_missing_count": means("entity_drift", "ghost_missing_count"),
            "evidence_coverage": means("evidence", "coverage"),
            "dead_ratio": means("dead_relations", "dead_ratio"),
            "source_cq_coverage": means("dead_relations", "source_cq_coverage"),
            "related_to_share": means("related_to", "share"),
        },
        "recurring_missing_in_master": dict(missing_names),
        "recurring_real_missing_in_master": dict(real_missing_names),
        "dead_by_type_total": dict(dead_types),
        "口径": "各样本 quality_observability 的 batch mean；warn-only，不进 gate",
    }


def load_master_entities_relations(master_path: str | Path | None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load (entities, relations) lists from a merged master file."""
    if not master_path:
        return [], []
    try:
        master = json.loads(Path(master_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    if not isinstance(master, dict):
        return [], []
    entities = master.get("nodes") or master.get("entities") or []
    relations = master.get("relationships") or master.get("relations") or []
    if isinstance(entities, dict):
        flat: List[Dict[str, Any]] = []
        for bucket in entities.values():
            if isinstance(bucket, list):
                flat.extend(x for x in bucket if isinstance(x, dict))
        entities = flat
    return (
        [e for e in entities if isinstance(e, dict)] if isinstance(entities, list) else [],
        [r for r in relations if isinstance(r, dict)] if isinstance(relations, list) else [],
    )
