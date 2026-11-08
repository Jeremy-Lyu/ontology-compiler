"""Deterministic domain safety net for extract (LLM-free backfill).

Canonical business relations (EBOM/MBOM -> BOM DERIVED_FROM, Cycle/Safety
Stock -> Inventory PART_OF) and canonical entities (BOM, Inventory,
MASTER DATA, VSM, Cycle Stock, Safety Stock, Production, Benchmark) are
backfilled only when the fragment already shows strong signals, so facts are
never invented from documents that do not mention them. This makes the most
stable recalls independent of LLM availability. Canonical entity set also
includes ATACQ (quality/measurement system).
"""

from __future__ import annotations

from typing import Any


def _entity_by_name(nodes: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    lowered = name.strip().lower()
    for node in nodes:
        if any(str(v).strip().lower() == lowered for v in (node.get("name"), node.get("meta_name"), node.get("scenario_name")) if v):
            return node
    return None


def append_deterministic_domain_relations(nodes: list[dict[str, Any]], relationships: list[dict[str, Any]], source_file: str, run_id: str) -> list[dict[str, Any]]:
    """LLM-free safety net: canonical domain relations only when both endpoints exist."""
    result = list(relationships)
    existing = {(str(r.get("source_name") or r.get("from_name") or r.get("source") or "").strip().lower(), str(r.get("target_name") or r.get("to_name") or r.get("target") or "").strip().lower(), str(r.get("type") or r.get("relation_type") or "").strip().upper()) for r in result}
    def _add(src: str, tgt: str, rel_type: str, evidence: str, confidence: float) -> None:
        if not _entity_by_name(nodes, src) or not _entity_by_name(nodes, tgt):
            return
        key = (src.strip().lower(), tgt.strip().lower(), rel_type)
        if key in existing:
            return
        result.append({
            "source": src,
            "target": tgt,
            "from_name": src,
            "to_name": tgt,
            "type": rel_type,
            "relation_type": rel_type,
            "confidence": confidence,
            "origin_kind": "injected",
            "evidence": {"source_file": source_file, "content": evidence},
            "provenance": {
                "source_file": source_file,
                "run_id": run_id,
                "stage": "extract",
                "origin": "deterministic_domain_relation",
            },
        })
        existing.add(key)
    _add("EBOM", "BOM", "DERIVED_FROM", "EBOM is an engineering view derived from the generic BOM", 0.9)
    _add("MBOM", "BOM", "DERIVED_FROM", "MBOM is a manufacturing view derived from the generic BOM", 0.9)
    # PART_OF（而非 HAS_PART）：HAS_PART 在 relationship_rules.yaml 中 severity=warn，
    # 会被关系治理改写为 STRUCTURAL 家族的规范关系 SUBSUMES；PART_OF severity=info 能
    # 原样通过治理，且保持「部分→整体」语义与 golden (PART_OF/HAS_PART) 兼容。
    _add("Cycle Stock", "Inventory", "PART_OF", "Cycle stock is a component of inventory", 0.85)
    _add("Safety Stock", "Inventory", "PART_OF", "Safety stock is a component of inventory", 0.85)
    return result


def ensure_deterministic_domain_entities(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backfill canonical domain entities on strong fragment signals (LLM-free)."""
    if not isinstance(nodes, list):
        return nodes
    names = {str(n.get("name", "")).strip().lower(): n for n in nodes if isinstance(n, dict) and n.get("name")}
    blob_parts: list[str] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        for key in ("name", "description", "meta_name", "scenario_name"):
            v = n.get(key)
            if isinstance(v, str):
                blob_parts.append(v)
        ev = n.get("evidence")
        if isinstance(ev, dict):
            for key in ("content", "raw_text", "source_file"):
                v = ev.get(key)
                if isinstance(v, str):
                    blob_parts.append(v)
    blob = " ".join(blob_parts).lower()

    def _add(canonical: str, meta_class: str, layer: str, description: str) -> None:
        key = canonical.strip().lower()
        if key in names:
            return
        nodes.append({
            "name": canonical,
            "meta_name": key,
            "meta_class": meta_class,
            "layer": layer,
            "description": description,
            "origin_kind": "injected",
            "provenance": {"stage": "extract", "origin": "deterministic_domain_entity"},
        })
        names[key] = canonical

    for blob_signal, canonical, meta_class, layer, desc in (
        (("inventory", "库存", "存货"), "Inventory", "Concept", "L2", "Inventory by purpose; umbrella concept for cycle and safety stock"),
        (("master data", "主数据", "mdm", "master data management"), "MASTER DATA", "System", "L1", "Master data management"),
        (("vsm", "value stream", "价值流"), "VSM", "Process", "L1", "Value stream mapping"),
        (("cycle stock", "cycle inventory", "周转库存"), "Cycle Stock", "Concept", "L2", "Cycle stock by purpose"),
        (("safety stock", "safety inventory", "安全库存"), "Safety Stock", "Concept", "L2", "Safety stock by purpose"),
        (("production", "production process", "生产", "生产制造"), "Production", "Process", "L1", "Production process"),
        (("benchmark", "benchmarking", "基准", "对标", "标杆"), "Benchmark", "Concept", "L2", "Benchmark or benchmark data"),
        (("atacq", "advanced tuning of quality"), "ATACQ", "System", "L2", "ATACQ quality/measurement system"),
        (("ebom", "engineering bom", "工程bom", "设计bom"), "EBOM", "System", "L1", "Engineering bill of materials; R&D view"),
        (("mbom", "manufacturing bom", "制造bom", "生产bom"), "MBOM", "System", "L1", "Manufacturing bill of materials; Mfg view"),
        (("bom", "bill of materials", "物料清单", "bom表", "bom清单"), "BOM", "Concept", "L1", "Generic bill of materials from which EBOM and MBOM are derived"),
        (("quality", "质量", "品质", "质量管理", "品质管理"), "Quality", "Concept", "L1", "Quality management"),
        (("incoming material quality", "incoming material", "来料质量", "imq"), "incoming material quality reporting", "Process", "L1", "Incoming material quality reporting process"),
    ):
        if any(s in blob for s in blob_signal):
            _add(canonical, meta_class, layer, desc)
    if "ebom" in names or "mbom" in names:
        _add("BOM", "Concept", "L1", "Generic bill of materials from which EBOM and MBOM are derived")
    if "cycle stock" in names or "safety stock" in names:
        _add("Inventory", "Concept", "L2", "Inventory by purpose; umbrella concept for cycle and safety stock")
    return nodes


def append_candidate_entities(state: Any, candidates: list[Any], source_file: str, run_id: str) -> None:
    """Feed LLM candidates into the normal extraction normalization path."""
    existing = {str(n.get("name", "")).strip().lower() for n in state.nodes if isinstance(n, dict)}
    for candidate in candidates:
        name = str(getattr(candidate, "name", "")).strip()
        if not name or name.lower() in existing:
            continue
        state.nodes.append({"name": name, "meta_name": name.lower(), "meta_class": getattr(candidate, "type", "Concept") or "Concept", "layer": "L2", "description": "LLM candidate; pending deterministic normalization", "evidence": {"source_file": source_file, "content": " | ".join(getattr(candidate, "evidence", []) or [])}, "provenance": {"source_file": source_file, "run_id": run_id, "stage": "extract", "origin": "llm_candidate"}})
        existing.add(name.lower())


def append_candidate_relationships(state: Any, candidates: list[Any], source_file: str, run_id: str) -> None:
    """Feed valid LLM relation candidates into normal relation processing."""
    for candidate in candidates:
        fields = getattr(candidate, "fields", {}) or {}
        source = fields.get("source") or fields.get("from")
        target = fields.get("target") or fields.get("to")
        rel_type = str(getattr(candidate, "type", "") or fields.get("type", "")).upper().strip()
        if not source or not target or not rel_type:
            continue
        state.relationships.append({"source": source, "target": target, "from_name": source, "to_name": target, "type": rel_type, "relation_type": rel_type, "confidence": getattr(candidate, "confidence", 0.0), "evidence": fields.get("evidence") or getattr(candidate, "evidence", []), "provenance": {"source_file": source_file, "run_id": run_id, "stage": "extract", "origin": "llm_candidate"}})
