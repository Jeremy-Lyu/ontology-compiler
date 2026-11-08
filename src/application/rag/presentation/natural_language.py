"""Presentation-layer renderers — natural_language (B-5, 2026-08-17).

Extracted from ``src.application.rag.formatting`` when splitting the 18 pure
renderer functions into report families under ``presentation/``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def synthesize_natural_answer(question: str, raw_answer: str, context: str) -> str:
    """Generate a natural-language synthesis from Dify's structured output.

    Takes the raw JSON (or already-formatted text) and produces a concise,
    natural Chinese answer that reads like an LLM comprehension — not a
    raw entity dump.
    """
    parsed: Any = None
    try:
        parsed = json.loads(raw_answer)
    except Exception:
        json_match = re.search(r'\{[\s\S]*\}', raw_answer)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except Exception:
                parsed = None

    if not isinstance(parsed, dict):
        return ""

    entity_layers = [
        k for k in ("L0_entities", "L1_entities", "L2_entities",
                    "L2_5_entities", "L3_entities")
        if k in parsed and isinstance(parsed.get(k), list)
    ]

    all_entities: list[dict[str, Any]] = []
    for layer in entity_layers:
        for ent in parsed[layer]:
            if isinstance(ent, dict):
                ent["_layer"] = layer.replace("_entities", "")
                all_entities.append(ent)

    rels = parsed.get("relationships")
    if not isinstance(rels, list):
        rels = []

    if not all_entities and not rels:
        return ""

    lines: list[str] = [
        f"## 基于本体知识的回答",
        "",
    ]

    layer_entities: dict[str, list[dict[str, Any]]] = {}
    for ent in all_entities:
        layer = ent.get("_layer", "unknown")
        layer_entities.setdefault(layer, []).append(ent)

    for layer in ("L0", "L1", "L2", "L2.5", "L3"):
        ents = layer_entities.get(layer, [])
        if not ents:
            continue
        lines.append(f"**{layer} 层关键实体：**")
        for ent in ents:
            name = ent.get("name", "?")
            desc = ent.get("description", "")
            eid = ent.get("meta_id", "")
            domain = ent.get("domain", "")
            cls = ent.get("meta_class", "")
            parts = [f"- **{name}**"]
            if cls:
                parts[0] += f"（{cls}"
                if domain:
                    parts[0] += f"·{domain}"
                parts[0] += "）"
            if desc:
                parts[0] += f" — {desc[:120]}"
            lines.append(parts[0])
        lines.append("")

    if rels:
        lines.append("**实体间关系：**")
        lines.append("")
        for r in rels:
            if not isinstance(r, dict):
                continue
            src = r.get("from", r.get("source", "?"))
            dst = r.get("to", r.get("target", "?"))
            rtype = r.get("type", r.get("relation", "related_to"))
            desc = r.get("description", "")
            line = f"- `{src}` **{rtype}** `{dst}`"
            if desc:
                line += f"：{desc[:100]}"
            lines.append(line)
        lines.append("")

    if all_entities or rels:
        lines.append("**综合分析：**")
        lines.append("")
        summary_bits = []
        for ent in all_entities[:3]:
            name = ent.get("name", "?")
            desc = ent.get("description", "")
            if desc:
                summary_bits.append(f"{name}是{desc}")
        if rels:
            rel_descs = [r.get("description", "") for r in rels if isinstance(r, dict) and r.get("description")]
            if rel_descs:
                summary_bits.append(
                    f"它们之间存在{len(rels)}种关系"
                    + (f"，包括{rels[0].get('type', '')}" if rels else "")
                )
        if summary_bits:
            lines.append("根据检索到的本体知识，" + "，".join(summary_bits) + "。")
        lines.append("")

    return "\n".join(lines)
def generate_natural_language_report(question: str, answer_text: str, context: str) -> str:
    """Generate natural language report when JSON parsing fails."""
    parts = [
        "## 深度业务分析报告",
        "",
        f"**分析主题**: {question}",
        "",
    ]

    if answer_text and len(answer_text) > 100:
        if "```" in answer_text:
            parts.append("**结构化结果**：")
            parts.append("")
            parts.append(answer_text[:3000])
        else:
            parts.append(answer_text[:3000])
    else:
        parts.append("基于检索到的本体知识，以下是相关分析：")
        parts.append("")
        parts.append(context[:2000])

    return "\n".join(parts)
