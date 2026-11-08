#!/usr/bin/env python3
"""将 governed packet 的知识块(chunks)导出为 Markdown,供 Gemini Notebook 使用。

用法:
    python scripts/export_governed_chunks_to_md.py [--out-dir outputs/export/gemini_notebook]

产物(按 chunk_type 分组 + 全量索引):
    <out_dir>/ontology_<type>.md   每个 chunk 渲染为一级条目
    <out_dir>/ontology_INDEX.md    全部条目标题 + chunk_id 的概览

说明:
    - 输入始终来自 default_governed_rag_dir() 指向的最新 governed packet
      (与 RAG 检索同一权威来源, 不会写入/污染 packet 目录)。
    - content 保留原文; 附加 title / parent_path / keywords 便于 NotebookLM 检索。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_chunks(rag_dir: Path) -> list[dict]:
    chunks_path = rag_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")
    rows = []
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _render_chunk(chunk: dict) -> list[str]:
    chunk_id = chunk.get("chunk_id", "unknown")
    chunk_type = chunk.get("chunk_type", "unknown")
    meta = chunk.get("meta", {}) or {}
    title = (meta.get("title") or "").strip() or chunk_id
    parent = " > ".join(meta.get("parent_path", []) or [])
    keywords = ", ".join(meta.get("keywords", []) or [])
    content = (chunk.get("content") or "").strip()

    lines = [f"## {title}  (`{chunk_id}`)", ""]
    meta_lines = [f"- 类型: {chunk_type}", f"- ID: {chunk_id}"]
    if parent:
        meta_lines.append(f"- 路径: {parent}")
    if keywords:
        meta_lines.append(f"- 关键词: {keywords}")
    lines.extend(meta_lines)
    lines.append("")
    lines.append(content)
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(_PROJECT_ROOT / "outputs" / "export" / "gemini_notebook"),
        help="导出目录 (默认 outputs/export/gemini_notebook)",
    )
    args = parser.parse_args()

    from src.application.rag.governed_packets import default_governed_rag_dir

    rag_dir = default_governed_rag_dir()
    print(f"✅ 使用 governed packet: {rag_dir}")
    chunks = _load_chunks(rag_dir)
    print(f"✅ 读取知识块: {len(chunks)} 条")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_type[chunk.get("chunk_type", "unknown")].append(chunk)

    index_lines = ["# Ontology 知识块索引", "", f"- 来源 packet: `{rag_dir.name}`", f"- 知识块总数: {len(chunks)}", ""]

    for chunk_type in sorted(by_type, key=lambda t: (-len(by_type[t]), t)):
        type_chunks = by_type[chunk_type]
        md_path = out_dir / f"ontology_{chunk_type}.md"
        lines = [
            f"# Ontology - {chunk_type} ({len(type_chunks)} 条)",
            "",
            f"> 来源 packet: `{rag_dir.name}` | 类型: {chunk_type}",
            "",
        ]
        for chunk in sorted(type_chunks, key=lambda c: c.get("chunk_id", "")):
            lines.extend(_render_chunk(chunk))
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"  ✓ {md_path.name}: {len(type_chunks)} 条")
        index_lines.append(f"## {chunk_type} ({len(type_chunks)} 条)")
        index_lines.append("")
        for chunk in sorted(type_chunks, key=lambda c: c.get("chunk_id", "")):
            meta = chunk.get("meta", {}) or {}
            title = (meta.get("title") or "").strip() or chunk.get("chunk_id", "")
            index_lines.append(f"- {title} (`{chunk.get('chunk_id')}`)")
        index_lines.append("")

    index_path = out_dir / "ontology_INDEX.md"
    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    print(f"  ✓ {index_path.name}")
    print(f"\n🎉 导出完成: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
