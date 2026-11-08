"""Presentation-layer renderers — sections_a_f (B-5, 2026-08-17).

Extracted from ``src.application.rag.formatting`` when splitting the 18 pure
renderer functions into report families under ``presentation/``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def build_section_a(
    target: Optional[Dict[str, Any]],
    related: Dict[str, List[Dict[str, Any]]],
    entity_map: Dict[str, Dict[str, Any]],
) -> str:
    """Build Section A: Data object definition (A-H framework report)."""
    parts = ["### A. 数据对象定义", ""]

    if not target:
        parts.append("根据检索到的本体知识，以下是相关实体的分析：\n")
        for eid, entity in list(entity_map.items())[:5]:
            name = entity.get("name", eid)
            desc = entity.get("description", "")
            parts.append(f"- **{name}**: {desc}")
        return "\n".join(parts)

    name = target.get("name", "未知实体")
    desc = target.get("description", "")
    meta_class = target.get("meta_class", "")
    domain = target.get("domain", "")
    entity_id = target.get("meta_id", "")

    internal_names = [name]
    if "BOM" in name.upper():
        internal_names.append("BOM")
        if "制造" in name or "Manufactur" in name:
            internal_names.append("MBOM")
        if "工程" in name or "Engineering" in name:
            internal_names.append("EBOM")
        if "物料需求" in name or "Material Requirement" in name:
            internal_names.append("MRP")

    parts.append(f"**{name}** (ID: {entity_id})")
    parts.append("")
    if desc:
        parts.append(f"> {desc}")
        parts.append("")

    if internal_names:
        parts.append(f"- **内部叫法**: {' / '.join(list(set(internal_names)))}")
    if meta_class:
        parts.append(f"- **数据分类**: {meta_class}")
    if domain:
        parts.append(f"- **所属领域**: {domain}")

    parts.append("")

    similar_entities = []
    for eid, entity in entity_map.items():
        if eid != entity_id:
            ename = entity.get("name", "")
            if any(kw in ename for kw in ["BOM", "MRP", "物料", "清单"]):
                similar_entities.append(entity)

    if similar_entities:
        parts.append("**核心区别**：")
        parts.append("")
        for se in similar_entities[:3]:
            sname = se.get("name", "")
            sdesc = se.get("description", "")
            if sdesc and desc:
                diff = compute_difference_hint(name, sname)
                if diff:
                    parts.append(f"- {diff}")

    return "\n".join(parts)
def build_section_b(
    target: Optional[Dict[str, Any]],
    related: Dict[str, List[Dict[str, Any]]],
    relationships: List[Dict[str, Any]],
    entity_map: Dict[str, Dict[str, Any]],
) -> str:
    """Build Section B: Processes and systems (A-H framework report)."""
    parts = ["### B. 涉及流程与系统", ""]

    if not target:
        return ""

    processes = related.get("contains", [])
    systems = related.get("uses", [])

    all_processes = []
    all_systems = []

    for rel_entry in relationships:
        from_id = rel_entry.get("from", "")
        to_id = rel_entry.get("to", "")
        rel_type = rel_entry.get("type", "").lower()

        for eid in [from_id, to_id]:
            if eid in entity_map:
                entity = entity_map[eid]
                eclass = entity.get("meta_class", "").lower()
                ename = entity.get("name", "")

                if "process" in eclass or "流程" in ename:
                    if entity not in all_processes:
                        all_processes.append(entity)
                elif "system" in eclass or "系统" in ename:
                    if entity not in all_systems:
                        all_systems.append(entity)

    if processes:
        for p in processes:
            entity = p["entity"]
            ename = entity.get("name", "")
            edesc = entity.get("description", "")
            parts.append(f"**{ename}**: {edesc}")
            parts.append("")

    if all_processes:
        parts.append("**关键流程步骤**：")
        parts.append("")
        for p in all_processes[:5]:
            ename = p.get("name", "")
            edesc = p.get("description", "")
            if edesc:
                parts.append(f"- {ename}: {edesc}")
            else:
                parts.append(f"- {ename}")

    if all_systems or systems:
        parts.append("")
        parts.append("**涉及系统**：")
        parts.append("")
        for s in (all_systems or [p["entity"] for p in systems[:5]]):
            sname = s.get("name", "")
            sdesc = s.get("description", "")
            if sdesc:
                parts.append(f"- **{sname}**: {sdesc}")
            else:
                parts.append(f"- **{sname}**")

    if not processes and not all_processes and not systems and not all_systems:
        parts.append("根据当前检索到的信息，该实体主要涉及以下关联：")
        parts.append("")
        for cat, entries in related.items():
            for entry in entries[:2]:
                entity = entry["entity"]
                ename = entity.get("name", "")
                rtype = entry.get("relationship", {}).get("type", "")
                parts.append(f"- 通过 **{rtype}** 关系关联到 **{ename}**")

    return "\n".join(parts)
def build_section_c(
    target: Optional[Dict[str, Any]],
    related: Dict[str, List[Dict[str, Any]]],
    entity_map: Dict[str, Dict[str, Any]],
) -> str:
    """Build Section C: Roles and decisions (A-H framework report)."""
    parts = ["### C. 角色与决策", ""]

    if not target:
        return ""

    actors = []
    for eid, entity in entity_map.items():
        eclass = entity.get("meta_class", "").lower()
        ename = entity.get("name", "")
        if "actor" in eclass or "角色" in ename or "人员" in ename or "师" in ename:
            actors.append(entity)

    if actors:
        parts.append("**相关角色**：")
        parts.append("")
        for actor in actors[:5]:
            aname = actor.get("name", "")
            adesc = actor.get("description", "")
            parts.append(f"- **{aname}**: {adesc}")

    parts.append("")
    parts.append("**决策影响**：")
    parts.append("")

    entity_name = target.get("name", "该实体")
    parts.append(f"- **生产指引**: 基于 {entity_name} 数据驱动生产执行")
    parts.append(f"- **采购拉动**: 基于 {entity_name} 结构生成采购需求")
    parts.append(f"- **成本核算**: 基于 {entity_name} 数据进行成本计算和分析")
    parts.append(f"- **库存优化**: 基于 {entity_name} 结构优化库存水平")

    return "\n".join(parts)
def compute_difference_hint(name1: str, name2: str) -> str:
    """Generate difference hint between two similar entities."""
    hints = []
    n1_lower = name1.lower()
    n2_lower = name2.lower()

    if "m" in n1_lower and "e" in n2_lower:
        hints.append(f"- **{name1}** 反映制造执行视角，按装配顺序组织；**{name2}** 反映设计视角，按功能系统组织")
    elif "e" in n1_lower and "m" in n2_lower:
        hints.append(f"- **{name1}** 反映设计视角，按功能系统组织；**{name2}** 反映制造执行视角，按装配顺序组织")

    return "\n".join(hints)
