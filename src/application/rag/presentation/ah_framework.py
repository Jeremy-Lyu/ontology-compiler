"""Presentation-layer renderers — ah_framework (B-5, 2026-08-17).

Extracted from ``src.application.rag.formatting`` when splitting the 18 pure
renderer functions into report families under ``presentation/``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Cross-family helpers (B-5): A-H framework report composes the dify JSON
# parser, the natural-language fallback, and the A/B/C section builders
# (D/E/F live in this file).
from .dify_structured import try_parse_dify_json  # noqa: F401
from .natural_language import generate_natural_language_report  # noqa: F401
from .sections_a_f import build_section_a, build_section_b, build_section_c  # noqa: F401


def identify_target_entity(
    question: str, l0_entities: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Identify the main entity being asked about."""
    if not l0_entities:
        return None

    question_lower = question.lower()
    best_match = None
    best_score = 0

    for entity in l0_entities:
        name = entity.get("name", "").lower()
        desc = entity.get("description", "").lower()
        meta_id = entity.get("meta_id", "").lower()

        score = 0
        if name and name in question_lower:
            score += 10
        if any(word in question_lower for word in name.split()):
            score += 5
        if desc and any(word in question_lower for word in desc.split()[:5]):
            score += 3
        if meta_id and meta_id.split("-")[-1] in question_lower:
            score += 2

        if score > best_score:
            best_score = score
            best_match = entity

    if best_match and best_score > 0:
        return best_match
    return l0_entities[0] if l0_entities else None
def collect_related_entities(
    target: Optional[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    entity_map: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect entities related to the target through relationships."""
    related = {"uses": [], "produces": [], "contains": [], "governs": [], "other": []}

    if not target:
        return related

    target_id = target.get("meta_id", "")

    for rel in relationships:
        from_id = rel.get("from", "")
        to_id = rel.get("to", "")
        rel_type = rel.get("type", "other").lower()

        if from_id == target_id and to_id in entity_map:
            entity = entity_map[to_id]
            entry = {"entity": entity, "relationship": rel}
            category = categorize_relation(rel_type)
            related[category].append(entry)

        if to_id == target_id and from_id in entity_map:
            entity = entity_map[from_id]
            entry = {"entity": entity, "relationship": rel}
            category = categorize_relation(rel_type)
            if category == "produces":
                category = "uses"
            related[category].append(entry)

    return related
def categorize_relation(rel_type: str) -> str:
    """Categorize relationship type."""
    rel_type_lower = rel_type.lower()
    if any(kw in rel_type_lower for kw in ["use", "utilize", "consume", "drive"]):
        return "uses"
    elif any(kw in rel_type_lower for kw in ["produce", "generat", "creat", "output"]):
        return "produces"
    elif any(kw in rel_type_lower for kw in ["contain", "includ", "has", "consist"]):
        return "contains"
    elif any(kw in rel_type_lower for kw in ["govern", "rule", "standard", "policy"]):
        return "governs"
    return "other"
def extract_business_context(context: str) -> str:
    """Extract business-relevant context for bonus section."""
    business_terms = [
        "CMP", "OHF", "BIS", "PLUS", "KDP", "SAP", "C-Note", "Allocation",
        "工位", "断点", "混合", "版本", "流程", "系统", "角色",
    ]

    found_terms = set()
    for term in business_terms:
        if term.lower() in context.lower():
            found_terms.add(term)

    if found_terms:
        return (
            f"\n\n**补充业务上下文**：\n"
            f"上下文中还包含以下业务相关信息：{', '.join(sorted(found_terms))}"
        )
    return ""
def generate_ah_framework_report(question: str, answer_text: str, context: str) -> str:
    """Generate A-H framework report from Dify's structured output."""
    parsed_data = try_parse_dify_json(answer_text)

    if not parsed_data:
        return generate_natural_language_report(question, answer_text, context)

    l0_entities = parsed_data.get("L0_entities", [])
    l1_entities = parsed_data.get("L1_entities", [])
    l2_entities = parsed_data.get("L2_entities", [])
    relationships = parsed_data.get("relationships", [])

    all_entities = (
        l0_entities + l1_entities + parsed_data.get("L2_5_entities", []) + l2_entities
    )

    if not all_entities:
        return generate_natural_language_report(question, answer_text, context)

    target_entity = identify_target_entity(question, l0_entities)

    entity_map = {}
    for e in all_entities:
        eid = e.get("meta_id", "")
        if eid:
            entity_map[eid] = e

    related_entities = collect_related_entities(
        target_entity, relationships, entity_map
    )

    sections = []
    sections.append(
        build_section_a(target_entity, related_entities, entity_map)
    )
    sections.append(
        build_section_b(target_entity, related_entities, relationships, entity_map)
    )
    sections.append(
        build_section_c(target_entity, related_entities, entity_map)
    )
    sections.append(
        build_section_d(target_entity, related_entities, entity_map)
    )
    sections.append(
        build_section_e(target_entity, related_entities, relationships, entity_map)
    )
    sections.append(
        build_section_f(target_entity, related_entities, entity_map)
    )

    report = "\n\n".join(filter(None, sections))

    if context and len(context) > 500:
        context_bonus = extract_business_context(context)
        if context_bonus:
            report += "\n\n" + context_bonus

    return report
def build_section_d(
    target: Optional[Dict[str, Any]],
    related: Dict[str, List[Dict[str, Any]]],
    entity_map: Dict[str, Dict[str, Any]],
) -> str:
    """Build Section D: Common issues and standards (A-H framework report)."""
    parts = ["### D. 常见问题与数据标准", ""]

    if not target:
        return ""

    rules = []
    for eid, entity in entity_map.items():
        eclass = entity.get("meta_class", "").lower()
        if "rule" in eclass or "standard" in eclass:
            rules.append(entity)

    if rules:
        parts.append("**相关规则和标准**：")
        parts.append("")
        for rule in rules[:4]:
            rname = rule.get("name", "")
            rdesc = rule.get("description", "")
            parts.append(f"- **{rname}**: {rdesc}")
    else:
        parts.append("**数据标准建议**：")
        parts.append("")
        parts.append("- **更新频率**: 必须与发布快照同步")
        parts.append("- **准确度**: 结构数据必须与实际状态100%对标")
        parts.append("- **稳定字段**: 核心标识、版本号、更新时间戳")
        parts.append("- **变更通知**: 相关团队应在变更后及时收到通知")

    return "\n".join(parts)
def build_section_e(
    target: Optional[Dict[str, Any]],
    related: Dict[str, List[Dict[str, Any]]],
    relationships: List[Dict[str, Any]],
    entity_map: Dict[str, Dict[str, Any]],
) -> str:
    """Build Section E: Typical business scenarios (A-H framework report)."""
    parts = ["### E. 典型业务场景", ""]

    if not target:
        return ""

    name = target.get("name", "")
    related_flows = []

    for rel in relationships[:3]:
        from_id = rel.get("from", "")
        to_id = rel.get("to", "")
        from_name = entity_map.get(from_id, {}).get("name", from_id)
        to_name = entity_map.get(to_id, {}).get("name", to_id)
        rel_type = rel.get("type", "关联")
        related_flows.append(
            f"{from_name} → [{rel_type}] → {to_name}"
        )

    if related_flows:
        parts.append(f"**{name} 典型使用流程**：")
        parts.append("")
        for flow in related_flows:
            parts.append(f"  {flow}")

    parts.append("")
    parts.append("**场景描述**：")
    parts.append("")
    parts.append(
        f"1. 数据接收: 从上游系统接收 {name} 结构数据"
    )
    parts.append(
        f"2. 数据转换: 根据业务规则进行格式转换和校验"
    )
    parts.append(
        f"3. 数据应用: 下游系统消费数据进行业务决策"
    )
    parts.append(
        f"4. 数据反馈: 将执行结果反馈给上游进行优化"
    )

    return "\n".join(parts)
def build_section_f(
    target: Optional[Dict[str, Any]],
    related: Dict[str, List[Dict[str, Any]]],
    entity_map: Dict[str, Dict[str, Any]],
) -> str:
    """Build Section F: Data sources and authority (A-H framework report)."""
    parts = ["### F. 数据来源与权威源", ""]

    if not target:
        return ""

    entity_id = target.get("meta_id", "")
    entity_name = target.get("name", "")

    parts.append(f"- **数据标识**: {entity_id}")
    parts.append(f"- **数据名称**: {entity_name}")
    parts.append(f"- **数据状态**: {target.get('status', 'active')}")
    parts.append(f"- **数据领域**: {target.get('domain', '未分类')}")
    parts.append("")

    parts.append("**权威源说明**：")
    parts.append("")
    parts.append(f"- 该数据由本体知识库中定义的权威源系统维护")
    parts.append(f"- 数据变更需遵循本体定义的审批流程")
    parts.append(f"- 冲突解决以实际执行数据为最终依据")

    return "\n".join(parts)
