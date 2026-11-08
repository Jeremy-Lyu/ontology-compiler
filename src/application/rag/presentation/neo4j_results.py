"""Presentation-layer renderers — neo4j_results (B-5, 2026-08-17).

Extracted from ``src.application.rag.formatting`` when splitting the 18 pure
renderer functions into report families under ``presentation/``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def format_neo4j_result(index: int, result: dict[str, Any]) -> str:
    """Format a single Neo4j search result as a context fragment."""
    node_type = result.get("node_type", "Unknown")
    name = result.get("name", "Unknown")
    desc = result.get("description", "")
    domain = result.get("domain", "")
    part = f"[{index}] {name} ({node_type}"
    if domain:
        part += f", 领域: {domain}"
    part += ")"
    if desc:
        desc_trunc = desc[:200] + "..." if len(desc) > 200 else desc
        part += f"\n    描述: {desc_trunc}"
    return part
def format_document_result(
    index: int,
    result: Any,
    *,
    max_chunk_length: int,
    l2_scenario_types: Iterable[str],
) -> str:
    """Format a single retrieved document chunk as a context fragment.

    Includes the RAG-OPT4 structured header (meta_class / aliases / parent /
    domain / score) injected from chunk metadata, with heuristic fallback that
    parses the first lines of chunk content when metadata is missing.
    """
    scenario_types = set(l2_scenario_types)
    chunk = result.chunk
    chunk_type = getattr(chunk, "chunk_type", "unknown")
    if chunk_type in scenario_types:
        content = chunk.content[:300]
    else:
        content = chunk.content[:max_chunk_length]
    source = chunk.meta.get("source", "unknown")

    meta = getattr(chunk, "meta", None) or {}
    # 优先用 meta 里显式字段；其次从 chunk.content 前 3 行做启发式解析
    # （ontology chunk 一般以 "meta_class: name\nID: xxx\n..." 开头）
    mcls = meta.get("meta_class") or meta.get("entity_type") or ""
    msub = meta.get("meta_subclass") or ""
    domain = meta.get("domain") or meta.get("sandbox_domain") or ""
    aliases_raw = meta.get("aliases") or meta.get("alias") or []
    if isinstance(aliases_raw, (list, tuple, set)):
        aliases = [str(x) for x in aliases_raw if str(x).strip()]
    elif isinstance(aliases_raw, str):
        aliases = [x.strip() for x in re.split(r"[,;，；、/|]", aliases_raw) if x.strip()]
    else:
        aliases = []
    pid = meta.get("parent_id") or meta.get("parent") or meta.get("parent_meta_id") or ""

    if not (mcls or domain):
        # 启发式：从 content 首 3 行里提取 "Xxx: yyy" 模式
        for line in (chunk.content or "").splitlines()[:3]:
            line_s = line.strip()
            if not mcls and (line_s.lower().startswith("class:") or line_s.lower().startswith("meta_class:")):
                mcls = line_s.split(":", 1)[1].strip()
            elif not domain and (line_s.lower().startswith("domain:") or line_s.lower().startswith("sandbox_domain:")):
                domain = line_s.split(":", 1)[1].strip()
            elif not pid and (line_s.lower().startswith("parent:") or line_s.lower().startswith("parent_id:")):
                pid = line_s.split(":", 1)[1].strip()

    # 拼结构化头部（如果有至少一个非空字段才加，避免浪费 context 字符）
    header_parts: list[str] = []
    if mcls:
        header_parts.append(f"类型={mcls}")
    if msub:
        header_parts.append(f"子类={msub}")
    if domain:
        header_parts.append(f"域={domain}")
    if aliases:
        header_parts.append(f"别名={','.join(aliases[:4])}")
    if pid:
        header_parts.append(f"父级ID={pid}")
    if isinstance(getattr(result, "score", None), (int, float)):
        header_parts.append(f"相关度={result.score:.3f}")

    header_line = ""
    if header_parts:
        header_line = "  [" + " | ".join(header_parts) + "]\n"

    chunk_id_str = getattr(chunk, "chunk_id", "")
    id_line = f"  ID: {chunk_id_str}\n" if chunk_id_str else ""

    return (
        f"[{index}] 来源: {source} (类型: {chunk_type})\n"
        + id_line
        + header_line
        + f"{content}\n"
    )
