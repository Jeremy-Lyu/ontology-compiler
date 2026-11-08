"""Presentation-layer renderers — dify_structured (B-5, 2026-08-17).

Extracted from ``src.application.rag.formatting`` when splitting the 18 pure
renderer functions into report families under ``presentation/``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def format_dify_structured_answer(raw_text: str, question: str) -> str:
    """Detect Dify structured-JSON output (L0_entities / relationships, etc.)
    and render it as a readable Chinese natural-language answer.

    Falls back to the raw text transparently if it does not look like the
    known ontology-analysis JSON schema.

    MAINTENANCE NOTE (2026-08-04): Adding new Dify workflow output fields
    ---------------------------------------------------------------------
    When the Dify Workflow app adds a NEW return field (e.g. a new entity
    layer like ``L4_entities`` or auxiliary data like ``quality_report``):
      1. Append the new key-name tuple-entry to ``_ENTITY_LAYER_KEYS`` below
         together with its Chinese display title in ``_LAYER_TITLES``.
      2. If the new field is NOT an entity-layer list but a scalar / dict
         auxiliary payload, add rendering logic in the dedicated
         "Auxiliary fields rendering" section (search for this keyword).
      3. Optionally promote the key in ``_dify_has_structured_shape`` keyset
         detection if it should alone trigger the structured-answer path.

    MAINTENANCE NOTE (2026-08-04): Empty-layer rendering
    -----------------------------------------------------
    Every declared layer that is PRESENT in the Dify JSON output is
    rendered explicitly, including with "（共 0 个）". This prevents the
    following anti-pattern: Dify's internal node-validation log lines
    (``L2_entities[] ❌ 空数组 ❌ 空数组 ❌ 空数组 100%`` style) leaking
    into the user-facing answer via the "raw_text fallback" path.
    """
    # =====================================================================
    # Step 0: scrub Dify workflow validation/status strings that are
    # occasionally concatenated into the answer text by upstream nodes.
    # Typical leakage patterns:
    #     "L2_entities[] ❌ 空数组 ❌ 空数组 ❌ 空数组 100%"
    #     "L2_5_entities[] ✅ 非空 92%"  /  "field_x ❌ missing 0%"
    #     Repeated "\nLx_entities[] ..." status banners
    # =====================================================================
    _status_strip_re = re.compile(
        r"(?:^|\n)[ \t]*(?:L\d(?:[_.]\d)?_entities\[\]|\b[A-Za-z_][\w\.]*\[\])"
        r"[^\n\r]*?(?:[❌✅⛔⚠️💯]|\b\d{1,3}\s*%\b)[^\n\r]*"
    )
    cleaned_text = _status_strip_re.sub("", raw_text)
    # Remove any leftover "X%\n" or "X %\n" suffixes attached to JSON digits
    cleaned_text = re.sub(
        r"(\d)\s*%\s*\n", r"\1\n", cleaned_text
    )
    cleaned_text = cleaned_text.strip()

    candidate = cleaned_text
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        first_nl = candidate.find("\n")
        if first_nl > 0 and candidate[:first_nl].strip().isalpha():
            candidate = candidate[first_nl:].strip()

    parsed: Any = None
    try:
        parsed = json.loads(candidate)
    except Exception:
        pass

    if not isinstance(parsed, dict):
        json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except Exception:
                parsed = None

    if not isinstance(parsed, dict):
        return cleaned_text if cleaned_text else raw_text

    keys = set(parsed.keys())

    # =====================================================================
    # ⬇⬇⬇  Adding new Dify App return fields? Edit these two maps  ⬇⬇⬇
    #
    # _ENTITY_LAYER_KEYS: ordered list of JSON keys each representing a
    #   list[{entity-dict}].  Order controls rendering order in the Markdown
    #   report.  If the Dify workflow adds a "L4_entities" / "Rules" layer
    #   just extend the tuple here.
    #
    # _LAYER_TITLES: Chinese human-readable heading for each layer key.
    # =====================================================================
    _ENTITY_LAYER_KEYS: Tuple[str, ...] = (
        "L0_entities",
        "L1_entities",
        "L2_entities",
        "L2_5_entities",
        "L3_entities",
        # ➕ append new layer keys returned by the Dify Workflow here
        # ➕ e.g.  "L4_entities",
        # ➕ e.g.  "Quality_entities",
    )
    _LAYER_TITLES: Dict[str, str] = {
        "L0_entities": "L0 抽象概念实体",
        "L1_entities": "L1 领域概念实体",
        "L2_entities": "L2 业务实体",
        "L2_5_entities": "L2.5 映射/规则实体",
        "L3_entities": "L3 系统实例实体",
        # ➕ add matching Chinese title for each new key added above
        # ➕ e.g.  "L4_entities": "L4 部署实例实体",
        # ➕ e.g.  "Quality_entities": "质量校验实体",
    }
    entity_layers = [k for k in _ENTITY_LAYER_KEYS if k in keys]
    _has_structured_shape = bool(entity_layers) or ("relationships" in keys)
    if not _has_structured_shape:
        return cleaned_text if cleaned_text else raw_text

    # --- Render structured output ---
    lines: List[str] = []
    lines.append(f"### 针对问题「{question}」的本体实体与关系分析")
    lines.append("")

    total_entities = 0
    # Explicitly render EVERY layer key that came back from Dify, even if
    # the list is empty.  This avoids leaving a "void" that accidentally
    # pulls Dify validation banners through the raw-text fallback.
    for layer in _ENTITY_LAYER_KEYS:
        if layer not in keys:
            # Layer was NOT produced by Dify at all -> skip silently
            continue
        items_raw = parsed.get(layer)
        items: List[Any]
        if isinstance(items_raw, list):
            items = items_raw
        elif items_raw is None:
            items = []
        else:
            # Not a list shape -> keep 0 count but log as-is later
            items = []
        total_entities += len(items)
        title = _LAYER_TITLES.get(layer, layer)
        lines.append(f"#### {title}（共 {len(items)} 个）")
        lines.append("")
        if not items:
            lines.append("_本层未返回实体（Dify Workflow 本次输出为空列表）_")
            lines.append("")
            continue
        for idx, ent in enumerate(items, 1):
            if not isinstance(ent, dict):
                continue
            mid = ent.get("meta_id", "—")
            mcls = ent.get("meta_class", "—")
            name = ent.get("name", ent.get("title", "—"))
            desc = ent.get("description") or ent.get("desc") or "（未提供描述）"
            domain = ent.get("domain", "—")
            status = ent.get("status", "—")
            parent = ent.get("parent_class") or ent.get("parent") or "—"
            evidence = ent.get("evidence") or ent.get("evidence_ref") or ""
            lines.append(f"{idx}. **{name}**")
            lines.append(f"   - ID: `{mid}` ｜ 类型: {mcls} ｜ 层级: {layer.replace('_entities','')} ｜ 域: {domain} ｜ 状态: {status}")
            if parent != "—":
                lines.append(f"   - 归属父类: {parent}")
            lines.append(f"   - 描述: {desc}")
            if evidence:
                lines.append(f"   - 证据: {evidence}")
            lines.append("")

    # --- Auxiliary fields rendering (non-layer payloads) ---
    # TODO: if Dify Workflow adds dict/list auxiliary payload keys, render
    # them here explicitly.  Example:
    #   if "quality_report" in keys: lines.append("#### 质量分析报告 ...")

    rels = parsed.get("relationships")
    if isinstance(rels, list) and rels:
        lines.append(f"#### 实体间关联关系（共 {len(rels)} 条）")
        lines.append("")
        for idx, r in enumerate(rels, 1):
            if not isinstance(r, dict):
                continue
            src = r.get("from") or r.get("source") or "?"
            dst = r.get("to") or r.get("target") or "?"
            rtype = r.get("type") or r.get("relation") or "related_to"
            desc = r.get("description") or r.get("desc") or ""
            ref = r.get("evidence_ref") or r.get("evidence") or ""
            conf = r.get("confidence")
            conf_str = f" ｜ 置信度: {conf}" if conf is not None else ""
            lines.append(f"{idx}. `{src}` ──**{rtype}**──▶ `{dst}`{conf_str}")
            if desc:
                lines.append(f"   - 说明: {desc}")
            if ref:
                lines.append(f"   - 证据: {ref}")
            lines.append("")
    elif "relationships" in keys:
        lines.append("#### 实体间关联关系（共 0 条）")
        lines.append("")
        lines.append("_本次未返回关联关系（Dify Workflow relationships 输出为空列表）_")
        lines.append("")

    is_empty_result = total_entities == 0 and (not isinstance(rels, list) or not rels)

    if is_empty_result:
        lines.append("> ⚠️ 本次分析未从提供的本体上下文中提取到有效实体或关系（所有 Dify 返回层均为空列表）。")
        lines.append("")
        lines.append("<details><summary>原始 Dify JSON 输出（已清洗状态标记）</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(parsed, ensure_ascii=False, indent=2)[:4000])
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    lines.append(f"_实体总计: {total_entities} ｜ 关系总计: {len(rels) if isinstance(rels, list) else 0} ｜ 分析模型: Dify_")
    rendered = "\n".join(lines)

    # Final safety scrub: ensure no stray status banners leak through
    rendered = _status_strip_re.sub("", rendered).strip()
    return rendered if rendered else cleaned_text or raw_text
def try_parse_dify_json(text: str) -> Dict[str, Any]:
    """Try to parse Dify's JSON output, handling various formats."""
    if not text:
        return {}

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "L0_entities" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    json_pattern = r'```json\s*\n(.*?)\n```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "L0_entities" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    json_pattern2 = r'\{[\s\S]*"L0_entities"[\s\S]*\}'
    match2 = re.search(json_pattern2, text)
    if match2:
        try:
            data = json.loads(match2.group(0))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    return {}
