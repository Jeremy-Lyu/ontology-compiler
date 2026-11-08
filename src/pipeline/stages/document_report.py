"""Document-level Markdown report generation stage.

After the export stage, scans pipeline extract fragments and generates:
- Per-source-document Markdown reports with YAML frontmatter
- A global inventory_summary.md index table
"""
from __future__ import annotations

import json
import re
import time
from argparse import Namespace
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import StageResult, PROJECT_ROOT, _read_json, _write_json, _entity_layer_value


def _find_fragments(pipeline_dir: Path, current_run_id: str) -> list[Path]:
    """Find all fragment JSON files under pipeline/<CURRENT_RUN_ID>/extract/files/.
    
    Only scans fragments from the current run, not all historical runs.
    This avoids regenerating all reports on every pipeline run.
    """
    fragments: list[Path] = []
    if not pipeline_dir.exists():
        return fragments
    # Only scan the current run directory
    current_run_dir = pipeline_dir / current_run_id
    extract_dir = current_run_dir / "extract" / "files"
    if not extract_dir.is_dir():
        return fragments
    for frag in extract_dir.rglob("fragment_*.json"):
        fragments.append(frag)
    return sorted(fragments)


def _extract_source_file(fragment: dict[str, Any]) -> str:
    """Extract source file path from fragment data."""
    for node in fragment.get("nodes", []):
        ev = node.get("evidence") or {}
        sf = ev.get("source_file") or node.get("source_file")
        if sf:
            return sf
    for inst in fragment.get("instances", []):
        ev = inst.get("evidence") or {}
        sf = ev.get("source_file") or inst.get("source_file")
        if sf:
            return sf
    return fragment.get("compilation_id") or fragment.get("run_id") or "unknown"


def _infer_domain(source_path: str) -> str:
    """Infer domain from source file path."""
    path_upper = source_path.upper()
    if "PPM" in path_upper or "PROJECT" in path_upper:
        return "项目与里程碑规划"
    if "BOM" in path_upper or "MBOM" in path_upper:
        return "制造工程与BOM"
    if "QUALITY" in path_upper or "ATACQ" in path_upper or "VIRA" in path_upper:
        return "质量与闭环管理"
    if "LOGISTIC" in path_upper or "SUPPLY" in path_upper:
        return "生产计划与供应链"
    return "通用"


def _build_summary(nodes: list[dict], instances: list[dict]) -> str:
    """Build a 150-char summary from L0/L1 entity descriptions."""
    descriptions: list[str] = []
    for n in nodes + instances:
        if str(n.get("layer", "")) in ("L0", "L1"):
            desc = n.get("description", "")
            name = n.get("name", "")
            if desc:
                descriptions.append(f"{name}: {desc}")
    summary = "; ".join(descriptions[:5])
    if len(summary) > 150:
        summary = summary[:147] + "..."
    return summary or "无可用摘要"


def _layer_distribution(nodes: list[dict], instances: list[dict]) -> dict[str, int]:
    """Count entities per layer."""
    dist: dict[str, int] = defaultdict(int)
    for n in nodes + instances:
        layer = _entity_layer_value(n) or "UNKNOWN"
        dist[layer] += 1
    return dict(sorted(dist.items()))


def _truncate(s: str, n: int = 80) -> str:
    """Truncate string for table display."""
    s = str(s).replace("|", "\\|").replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s


def _generate_document_md(
    doc_id: str,
    fragment: dict[str, Any],
    source_file: str,
    fragment_path: Path,
) -> str:
    """Generate a single document Markdown report."""
    nodes = fragment.get("nodes", [])
    instances = fragment.get("instances", [])
    relationships = fragment.get("relationships", [])
    run_id = fragment.get("run_id", "")
    quality_level = fragment.get("quality_level", "N/A")
    quality_score = fragment.get("quality_score", 0)
    validation_passed = fragment.get("validation_passed", False)

    # Build meta_id -> name mapping for lookup
    meta_id_to_name: dict[str, str] = {}
    for node in nodes:
        meta_id = node.get("meta_id", "")
        name = node.get("name", "")
        if meta_id and name:
            meta_id_to_name[meta_id] = name

    layer_dist = _layer_distribution(nodes, instances)
    domain = _infer_domain(source_file)
    summary = _build_summary(nodes, instances)
    extracted_date = datetime.now().strftime("%Y-%m-%d")
    if run_id and "-" in run_id:
        parts = run_id.split("-")
        if len(parts) >= 2:
            date_part = parts[1] if len(parts) > 1 else ""
            if len(date_part) == 8 and date_part.isdigit():
                extracted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"

    anchor_entities = [
        {"meta_id": n.get("meta_id", ""), "name": n.get("name", ""),
         "layer": n.get("layer", ""), "meta_class": n.get("meta_class", "")}
        for n in nodes[:5]
    ]

    # YAML frontmatter
    yaml_lines = [
        "---",
        f'doc_id: "{doc_id}"',
        f'source_file: "{Path(source_file).name}"',
        f'source_path: "{source_file}"',
        f'run_id: "{run_id}"',
        f'extracted_date: "{extracted_date}"',
        f'domain_l1: "{domain}"',
        f'quality_level: "{quality_level}"',
        f'quality_score: {round(quality_score, 2) if isinstance(quality_score, (int, float)) else 0}',
        f'entity_count: {len(nodes)}',
        f'instance_count: {len(instances)}',
        f'relationship_count: {len(relationships)}',
        "layer_distribution:",
    ]
    for layer, count in layer_dist.items():
        yaml_lines.append(f"  {layer}: {count}")
    yaml_lines.append("anchor_entities:")
    for ae in anchor_entities:
        yaml_lines.append(f"  - meta_id: \"{ae['meta_id']}\"")
        yaml_lines.append(f"    name: \"{ae['name']}\"")
        yaml_lines.append(f"    layer: \"{ae['layer']}\"")
        yaml_lines.append(f"    meta_class: \"{ae['meta_class']}\"")
    yaml_lines.append(f'summary: "{_truncate(summary, 150)}"')
    yaml_lines.append("---")

    # Markdown body
    body_lines = [
        "",
        f"# {Path(source_file).name} — 知识抽取报告",
        "",
        "## 1. 业务摘要",
        summary,
        "",
        "## 2. 关键实体矩阵",
        "| 实体ID | 名称 | 层级 | 类别 | 描述 |",
        "|--------|------|------|------|------|",
    ]
    for n in nodes[:30]:
        body_lines.append(
            f"| {n.get('meta_id', '')} | {_truncate(n.get('name', ''))} | "
            f"{n.get('layer', '')} | {n.get('meta_class', '')} | "
            f"{_truncate(n.get('description', ''))} |"
        )
    if len(nodes) > 30:
        body_lines.append(f"| ... | ({len(nodes) - 30} more) | | | |")

    body_lines.extend([
        "",
        "## 3. 实体间关系",
        "| 源实体 | 关系类型 | 目标实体 |",
        "|--------|---------|---------|",
    ])
    for rel in relationships[:30]:
        # Get from entity name
        from_val = rel.get('from_name') or ''
        if not from_val:
            from_id = rel.get('from', '')
            from_val = meta_id_to_name.get(from_id, from_id)
        # Get to entity name
        to_val = rel.get('to_name') or ''
        if not to_val:
            to_id = rel.get('to', '')
            to_val = meta_id_to_name.get(to_id, to_id)
        # Get relation type
        rel_type = rel.get('type') or rel.get('relation_type') or ''
        body_lines.append(
            f"| {_truncate(from_val)} | {_truncate(rel_type)} | {_truncate(to_val)} |"
        )
    if len(relationships) > 30:
        body_lines.append(f"| ... | ({len(relationships) - 30} more) | ... |")

    body_lines.extend([
        "",
        "## 4. 证据溯源",
        "| 实体ID | 来源文件 | 行号范围 | 证据内容摘要 |",
        "|--------|---------|---------|-------------|",
    ])
    for n in nodes[:15]:
        ev = n.get("evidence") or {}
        ev_file = Path(ev.get("source_file", "")).name
        line_range = ev.get("line_range", "")
        content = _truncate(str(ev.get("content", "")), 100)
        if ev_file or content:
            body_lines.append(
                f"| {n.get('meta_id', '')} | {ev_file} | {line_range} | {content} |"
            )

    body_lines.extend([
        "",
        "## 5. 抽取质量评估",
        f"- 质量等级: {quality_level}",
        f"- 质量评分: {round(quality_score, 2) if isinstance(quality_score, (int, float)) else 'N/A'}",
        f"- 验证状态: {'通过' if validation_passed else '未通过'}",
        f"- 警告数: {len(fragment.get('warnings', []))}",
    ])

    return "\n".join(yaml_lines + body_lines) + "\n"


import logging
logger = logging.getLogger(__name__)

def run_document_report(args: Namespace) -> StageResult:
    """Generate per-document Markdown reports from extract fragments.
    
    Only generates reports for fragments processed in the current pipeline run,
    not all historical documents. This significantly improves performance.
    Handles permission errors gracefully by skipping files that can't be written.
    """
    start = time.time()

    ontology_base_dir = PROJECT_ROOT / "outputs" / "runs" / "ontology"
    pipeline_dir = ontology_base_dir / "pipeline"
    reports_dir = ontology_base_dir / "reports" / "documents"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Get current run_id from args
    current_run_id = getattr(args, "run_id", "")
    if not current_run_id:
        return StageResult(
            "document_report", True, "no run_id provided (skipped)",
            {"reports_generated": 0}, round(time.time() - start, 3),
        )

    fragments = _find_fragments(pipeline_dir, current_run_id)
    if not fragments:
        return StageResult(
            "document_report", True, "no fragments found in current run (skipped)",
            {"reports_generated": 0}, round(time.time() - start, 3),
        )

    # Group fragments by source_file
    grouped: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for frag_path in fragments:
        data = _read_json(frag_path, {})
        if not data:
            continue
        source_file = _extract_source_file(data)
        grouped[source_file].append((frag_path, data))

    # Generate per-document reports
    inventory_rows: list[dict[str, Any]] = []
    skipped_count = 0
    errors: list[str] = []
    doc_counter = 0
    for source_file, frag_list in sorted(grouped.items()):
        doc_counter += 1
        doc_id = f"KB-ONT-{doc_counter:04d}"

        # Use the fragment with highest quality_score
        best_path, best_data = max(frag_list, key=lambda x: x[1].get("quality_score", 0))

        md_content = _generate_document_md(doc_id, best_data, source_file, best_path)
        safe_name = re.sub(r'[^\w\-.]', '_', Path(source_file).stem)[:80]
        md_path = reports_dir / f"{doc_id}_{safe_name}.md"

        # Try to write the file, handle permission errors gracefully
        try:
            md_path.write_text(md_content, encoding="utf-8")
        except PermissionError:
            logger.warning(f"[document_report] Permission denied when writing {md_path}, skipping")
            skipped_count += 1
            errors.append(f"Permission denied: {md_path}")
            continue
        except OSError as e:
            logger.warning(f"[document_report] OS error when writing {md_path}: {e}, skipping")
            skipped_count += 1
            errors.append(f"OS error: {md_path}: {e}")
            continue

        nodes = best_data.get("nodes", [])
        instances = best_data.get("instances", [])
        relationships = best_data.get("relationships", [])
        inventory_rows.append({
            "doc_id": doc_id,
            "source_file": Path(source_file).name,
            "entity_count": len(nodes),
            "relationship_count": len(relationships),
            "quality_level": best_data.get("quality_level", "N/A"),
            "extracted_date": datetime.now().strftime("%Y-%m-%d"),
        })

    # Generate inventory_summary.md - always regenerate because it's the global index
    inv_lines = [
        "# 知识资产大账本 (Inventory Summary)",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总文档数 (历史 + 当前运行): {len(inventory_rows)}",
        f"本次运行新增/更新: {len(inventory_rows)}",
        f"本次跳过（权限错误）: {skipped_count}",
        "",
        "| 文档编号 | 源文档名称 | 实体数 | 关系数 | 质量等级 | 抽取日期 |",
        "|---------|----------|-------|-------|---------|---------|",
    ]
    for row in inventory_rows:
        inv_lines.append(
            f"| {row['doc_id']} | {row['source_file']} | "
            f"{row['entity_count']} | {row['relationship_count']} | "
            f"{row['quality_level']} | {row['extracted_date']} |"
        )

    inv_path = ontology_base_dir / "reports" / "inventory_summary.md"
    # Try to write inventory summary, if it fails, just log and continue
    try:
        inv_path.write_text("\n".join(inv_lines) + "\n", encoding="utf-8")
    except PermissionError:
        logger.warning(f"[document_report] Permission denied when writing inventory summary {inv_path}")
        errors.append(f"Permission denied: {inv_path}")
    except OSError as e:
        logger.warning(f"[document_report] OS error when writing inventory summary: {e}")
        errors.append(f"OS error: {inv_path}: {e}")

    generated_count = len(inventory_rows)
    message = f"generated {generated_count} document reports, skipped {skipped_count}"
    return StageResult(
        "document_report",
        len(errors) < generated_count,  # Only fail if all generated failed
        message,
        {
            "reports_generated": generated_count,
            "reports_skipped": skipped_count,
            "reports_dir": str(reports_dir),
            "inventory_path": str(inv_path),
            "errors": errors,
        },
        round(time.time() - start, 3),
    )
