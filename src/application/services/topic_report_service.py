"""Generate TBH topic full report with run-scoped + shared outputs.

Moved from scripts/maintenance/generate_tbh_topic_report.py to eliminate
cross-layer import (service layer importing scripts).

The script layer should remain a thin CLI wrapper around this module.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ontology_compiler.config import get_runtime_paths
from core.utils.file_helpers import load_json_optional


def _load_json(path: Path) -> dict[str, Any] | None:
    payload = load_json_optional(path)
    return payload if isinstance(payload, dict) else None


def _resolve_run_dir(
    run_id: str | None = None,
    run_dir: str | None = None,
    *,
    runs_root: Path,
    output_root: Path,
) -> Path:
    if run_dir:
        return Path(run_dir)
    if run_id:
        return runs_root / run_id
    if not runs_root.exists():
        return output_root

    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    return run_dirs[-1] if run_dirs else output_root


def _collect_summary(
    tbh_supplement: dict[str, Any] | None,
    layer2_instances: dict[str, Any] | None,
) -> dict[str, int]:
    supplement = tbh_supplement or {}
    layer2 = layer2_instances or {}

    return {
        "business_capabilities": len(supplement.get("nodes", [])),
        "systems": len(supplement.get("instances", [])),
        "processes": len(supplement.get("relationships", [])),
        "rules": _count_sandbox_rules(layer2),
    }


def _count_sandbox_rules(layer2: dict[str, Any]) -> int:
    rules = 0
    for domain_data in (layer2.get("domains") or {}).values():
        rules += len(domain_data.get("sandbox_rules", []))
    return rules


def _render_report(
    data_dir: Path,
    summary: dict[str, int],
    tbh_supplement: dict[str, Any] | None,
    analysis_manifest: dict[str, Any] | None = None,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    supplement = tbh_supplement or {}
    source_files = _collect_source_files(supplement)
    quality = analysis_manifest.get("quality", {}) if analysis_manifest else {}
    domain, topic_id = _extract_manifest_identity(analysis_manifest)
    object_types = _count_object_types(supplement)
    relationship_types = _count_relationship_types(supplement)
    key_systems = _extract_key_systems(supplement)
    warnings = quality.get("warnings", [])

    lines: list[str] = []
    lines.extend(_build_report_header(ts, data_dir))
    lines.extend(_build_overview_section(domain, topic_id, quality))
    lines.extend(_build_key_metrics_section(summary))
    lines.extend(_build_distribution_section("## 对象类型分布", object_types))
    lines.extend(_build_distribution_section("## 关系类型分布", relationship_types))
    lines.extend(_build_key_systems_section(key_systems))
    lines.extend(_build_source_files_section(source_files))
    if warnings:
        lines.extend(_build_warnings_section(warnings))
    lines.append("---")
    lines.append(f"*报告生成时间: {ts}*")
    return "\n".join(lines)


def _collect_source_files(supplement: dict[str, Any]) -> set[str]:
    source_files = set()
    for node in supplement.get("nodes", []):
        src = node.get("source_file")
        if src:
            source_files.add(Path(src).name)
    for inst in supplement.get("instances", []):
        src = inst.get("instance_source") or inst.get("source_file")
        if src:
            source_files.add(Path(src).name)
    for rel in supplement.get("relationships", []):
        src = rel.get("source_file")
        if src:
            source_files.add(Path(src).name)
    return source_files


def _extract_manifest_identity(analysis_manifest: dict[str, Any] | None) -> tuple[str, str]:
    domain = analysis_manifest.get("domain", "N/A") if analysis_manifest else "N/A"
    topic_id = analysis_manifest.get("topic_id", "N/A") if analysis_manifest else "N/A"
    return domain, topic_id


def _count_object_types(supplement: dict[str, Any]) -> dict[str, int]:
    object_types: dict[str, int] = {}
    for node in supplement.get("nodes", []):
        obj_class = node.get("meta_class", "Unknown")
        object_types[obj_class] = object_types.get(obj_class, 0) + 1
    return object_types


def _count_relationship_types(supplement: dict[str, Any]) -> dict[str, int]:
    relationship_types: dict[str, int] = {}
    for rel in supplement.get("relationships", []):
        predicate = rel.get("predicate", "Unknown")
        relationship_types[predicate] = relationship_types.get(predicate, 0) + 1
    return relationship_types


def _extract_key_systems(supplement: dict[str, Any]) -> list[str]:
    systems = [node for node in supplement.get("nodes", []) if node.get("meta_class") == "System"]
    return [s.get("meta_name", s.get("meta_id", "Unknown")) for s in systems[:5]]


def _build_report_header(ts: str, data_dir: Path) -> list[str]:
    return [
        "# TBH 专题全量知识报告",
        f"**生成时间**: {ts}",
        f"**数据目录**: `{data_dir}`",
        "",
    ]


def _build_overview_section(domain: str, topic_id: str, quality: dict[str, Any]) -> list[str]:
    return [
        "## 整体概览",
        f"- **领域**: {domain}",
        f"- **主题ID**: {topic_id}",
        f"- **质量等级**: {quality.get('grade', 'N/A')}",
        f"- **L1质量分数**: {quality.get('l1_score', 'N/A')}",
        f"- **L2质量分数**: {quality.get('l2_score', 'N/A')}",
        f"- **警告数**: {len(quality.get('warnings', []))}",
        "",
    ]


def _build_key_metrics_section(summary: dict[str, int]) -> list[str]:
    return [
        "## 关键指标",
        f"- **业务能力（nodes）**: {summary['business_capabilities']}",
        f"- **实例（instances）**: {summary['systems']}",
        f"- **关系（relationships）**: {summary['processes']}",
        f"- **规则（sandbox_rules）**: {summary['rules']}",
        "",
    ]


def _build_distribution_section(title: str, counts: dict[str, int]) -> list[str]:
    lines = [title]
    for item, count in sorted(counts.items()):
        lines.append(f"- **{item}**: {count}")
    lines.append("")
    return lines


def _build_key_systems_section(key_systems: list[str]) -> list[str]:
    lines = ["## 关键系统（前5个）"]
    if key_systems:
        lines.extend(f"- {sys}" for sys in key_systems)
    else:
        lines.append("- （未检测到系统）")
    lines.append("")
    return lines


def _build_source_files_section(source_files: set[str]) -> list[str]:
    lines = ["## 来源文件"]
    if source_files:
        lines.extend(f"- {name}" for name in sorted(source_files))
    else:
        lines.append("- （未检测到来源文件）")
    lines.append("")
    return lines


def _build_warnings_section(warnings: list[str]) -> list[str]:
    lines = ["## 警告信息"]
    lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return lines


def generate_topic_report_for_run(
    run_id: str | None = None,
    run_dir: str | None = None,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Generate report for a run and write shared + run-scoped report files."""
    root = project_root or Path.cwd()
    paths = get_runtime_paths()
    output_root = root / paths.get("outputs_ontology_root", "outputs/ontology")
    runs_root = output_root / "runs"

    data_dir = _resolve_run_dir(
        run_id=run_id, run_dir=run_dir, runs_root=runs_root, output_root=output_root
    )

    tbh_supplement = _load_json(data_dir / "tbh_supplement.json")
    layer2_instances = _load_json(data_dir / "layer2_instances.json")
    analysis_manifest = _load_json(data_dir / "analysis_manifest.json")

    if not tbh_supplement and not layer2_instances:
        return {
            "generated": False,
            "status": "ERROR",
            "error": "未找到 TBH 数据文件，请先运行 TBH 分析",
            "data_dir": str(data_dir),
        }

    summary = _collect_summary(tbh_supplement, layer2_instances)
    report_content = _render_report(data_dir, summary, tbh_supplement, analysis_manifest)

    shared_output = output_root / "TBH_TOPIC_FULL_REPORT.md"
    shared_output.parent.mkdir(parents=True, exist_ok=True)
    with open(shared_output, "w", encoding="utf-8") as f:
        f.write(report_content)

    run_output = data_dir / "TBH_TOPIC_FULL_REPORT.md"
    run_output.parent.mkdir(parents=True, exist_ok=True)
    with open(run_output, "w", encoding="utf-8") as f:
        f.write(report_content)

    return {
        "generated": True,
        "status": "SUCCESS",
        "data_dir": str(data_dir),
        "shared_output": str(shared_output),
        "run_output": str(run_output),
        "summary": summary,
    }
