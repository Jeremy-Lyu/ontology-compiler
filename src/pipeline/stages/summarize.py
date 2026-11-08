"""Summarize stage."""
import json
import time
import shutil
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import StageResult, PROJECT_ROOT, _read_json, _write_json, _latest_file

try:
    from ontology_compiler.services.quality_assurance.unknown_governance import (
        UnknownGovernanceService,
    )
    _UNKNOWN_GOV_AVAILABLE = True
except ImportError:
    _UNKNOWN_GOV_AVAILABLE = False


def _collect_run_artifacts(run_dir: Path) -> dict[str, Any]:
    files_dir = run_dir / "files"
    snapshots: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    artifact_manifest: list[dict[str, Any]] = []
    
    def _artifact_kind(path: Path) -> str:
        name = path.name.lower()
        if name.endswith("analysis_snapshot.json"):
            return "analysis_snapshot"
        if name.endswith("signal_catalog.json"):
            return "signal_catalog"
        if name.endswith("trace_index.json"):
            return "trace_index"
        if name.endswith("ontology_delta.json"):
            return "ontology_delta"
        if name.endswith("compilation_result.json"):
            return "compilation_result"
        if name.endswith("manifest.json"):
            return "manifest"
        if name.endswith("summary_report.json"):
            return "summary_report"
        if name.endswith("pipeline_summary"):
            return "pipeline_summary"
        if path.suffix.lower() == ".json":
            return "json"
        if path.suffix.lower() in {".md", ".markdown"}:
            return "markdown"
        return path.suffix.lower().lstrip(".") or "file"
    
    def _artifact_stage(path: Path) -> str:
        parts = list(path.parts)
        if "files" in parts:
            return "extract"
        name = path.name.lower()
        if name.startswith("ontology_audit_report"):
            return "audit"
        if name.startswith("pipeline_summary") or name.startswith("run_index") or name.startswith("summary_report"):
            return "summarize"
        if name.endswith("manifest.json"):
            return "extract"
        if "fragment_" in name:
            return "extract"
        if name.endswith("merged_fragment.json"):
            return "merge"
        if name.endswith("master_ontology.json"):
            return "merge"
        if name.endswith("relationships.json") or "layers" in parts:
            return "export"
        return "unknown"
    
    def _append_artifact(path: Path) -> None:
        if not path.is_file():
            return
        try:
            stat = path.stat()
            artifact_manifest.append({
                "path": str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path),
                "abs_path": str(path),
                "name": path.name,
                "kind": _artifact_kind(path),
                "stage": _artifact_stage(path),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except Exception:
            artifact_manifest.append({
                "path": str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path),
                "abs_path": str(path),
                "name": path.name,
                "kind": _artifact_kind(path),
                "stage": _artifact_stage(path),
                "size_bytes": None,
                "modified_at": None,
            })
    
    if files_dir.exists():
        for snapshot_path in files_dir.glob("*/analysis_snapshot.json"):
            snapshot = _read_json(snapshot_path, {})
            if snapshot:
                snapshots.append(snapshot)
            _append_artifact(snapshot_path)
        for signal_path in files_dir.glob("*/signal_catalog.json"):
            signal = _read_json(signal_path, {})
            if signal:
                signals.append(signal)
            _append_artifact(signal_path)
        for trace_path in files_dir.glob("*/trace_index.json"):
            trace = _read_json(trace_path, {})
            if trace:
                traces.append(trace)
            _append_artifact(trace_path)
    
    for top_level in run_dir.glob("*.json"):
        _append_artifact(top_level)
    for top_level in run_dir.glob("*.md"):
        _append_artifact(top_level)
    
    schema_refs = sorted({
        ref
        for payload in snapshots + signals + traces
        for ref in (payload.get("schemas_used") or payload.get("schemas") or payload.get("schema_refs") or [])
        if isinstance(ref, str) and ref.strip()
    })
    
    skill_sources: dict[str, list[str]] = {}
    observed_pairs: list[dict[str, Any]] = []
    direct_frequency: dict[str, int] = {}
    
    def _track(bucket: dict[str, int], skill_name: str, source: str, source_type: str) -> None:
        bucket[skill_name] = bucket.get(skill_name, 0) + 1
        skill_sources.setdefault(skill_name, [])
        if source not in skill_sources[skill_name]:
            skill_sources[skill_name].append(source)
        observed_pairs.append({"skill": skill_name, "source": source, "source_type": source_type})
    
    for signal_path in files_dir.glob("*/signal_catalog.json") if files_dir.exists() else []:
        signal = _read_json(signal_path, {})
        if not isinstance(signal, dict):
            continue
        source_label = str(signal.get("file") or signal_path.parent.name)
        for skill in signal.get("skills") or []:
            if isinstance(skill, str) and skill.strip():
                _track(direct_frequency, skill.strip(), source_label, "signal")
    
    artifact_manifest = sorted(artifact_manifest, key=lambda item: (item.get("stage") or "", item.get("kind") or "", item.get("path") or ""))
    artifact_index_by_stage: dict[str, list[dict]] = {}
    artifact_index_by_kind: dict[str, list[dict]] = {}
    for artifact in artifact_manifest:
        artifact_index_by_stage.setdefault(str(artifact.get("stage") or "unknown"), []).append(artifact)
        artifact_index_by_kind.setdefault(str(artifact.get("kind") or "file"), []).append(artifact)
    
    return {
        "available": True,
        "run_dir": str(run_dir),
        "snapshot_count": len(snapshots),
        "signal_count": len(signals),
        "trace_count": len(traces),
        "schema_refs": schema_refs,
        "skill_refs": sorted(direct_frequency.keys()),
        "artifact_manifest": artifact_manifest,
        "artifact_index": {
            "by_stage": artifact_index_by_stage,
            "by_kind": artifact_index_by_kind,
            "count": len(artifact_manifest),
        },
        "skill_lineage": {
            "direct": {"skills": sorted(direct_frequency.keys()), "frequency": direct_frequency},
            "skill_sources": skill_sources,
            "observed_pairs": observed_pairs[:200],
        },
    }


def _entity_layer_value(entity: dict) -> str:
    if not isinstance(entity, dict):
        return "unknown"
    if "layer" in entity:
        return str(entity["layer"])
    meta_id = str(entity.get("meta_id", "") or entity.get("entity_id", "") or "")
    if meta_id.startswith("L0"):
        return "L0"
    elif meta_id.startswith("L1"):
        return "L1"
    elif meta_id.startswith("L25"):
        return "L2.5"
    elif meta_id.startswith("L3"):
        return "L3"
    elif meta_id.startswith("L2"):
        return "L2"
    return "L1"


def run_summarize(args: Namespace) -> StageResult:
    start = time.time()
    master_path = Path(args.master_path)
    if not master_path.exists():
        return StageResult("summarize", False, f"master not found: {master_path}", elapsed_seconds=round(time.time() - start, 3))
    
    master = _read_json(master_path, {})
    
    nodes = master.get("nodes", [])
    instances = master.get("instances", [])
    relationships = master.get("relationships") or master.get("edges") or []
    
    metadata = master.get("metadata", {}) if isinstance(master.get("metadata", {}), dict) else {}
    rel_stats = metadata.get("relationship_stats", {}) if isinstance(metadata.get("relationship_stats", {}), dict) else {}
    rel_validation_issues = metadata.get("relationship_validation_issues", []) if isinstance(metadata.get("relationship_validation_issues", []), list) else []
    
    layer_distribution = {}
    all_entities = nodes + [item for item in instances if isinstance(item, dict)]
    for n in all_entities:
        layer = _entity_layer_value(n)
        layer_distribution[layer] = layer_distribution.get(layer, 0) + 1
    
    relationship_layer_distribution = {}
    node_id_map = {}
    for n in all_entities:
        eid = str(n.get("meta_id") or n.get("entity_id") or n.get("id") or "")
        if eid:
            node_id_map[eid] = _entity_layer_value(n)
    
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        source = rel.get("source") or rel.get("from")
        target = rel.get("target") or rel.get("to")
        
        if isinstance(source, dict):
            source_layer = _entity_layer_value(source)
        else:
            source_id = str(source or "")
            source_layer = node_id_map.get(source_id, "unknown")
        
        if isinstance(target, dict):
            target_layer = _entity_layer_value(target)
        else:
            target_id = str(target or "")
            target_layer = node_id_map.get(target_id, "unknown")
        
        layer_key = f"{source_layer}->{target_layer}"
        relationship_layer_distribution[layer_key] = relationship_layer_distribution.get(layer_key, 0) + 1
    
    run_dir: Path | None = None
    if getattr(args, "run_dir", ""):
        candidate = Path(args.run_dir)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            run_dir = candidate
    
    if run_dir is None:
        candidates = [p for p in Path(args.output_dir).glob("UNIFIED-*") if p.is_dir()]
        if candidates:
            run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    
    artifacts = {}
    if run_dir and run_dir.exists():
        artifacts = _collect_run_artifacts(run_dir)
    
    unknown_gov_report = {}
    if _UNKNOWN_GOV_AVAILABLE:
        try:
            gov_service = UnknownGovernanceService()
            all_entities_for_gov = nodes + [item for item in instances if isinstance(item, dict)]
            stats, alerts = gov_service.analyze(all_entities_for_gov, relationships)
            unknown_gov_report = gov_service.generate_report(stats, alerts)
        except Exception as e:
            unknown_gov_report = {"error": str(e)}
    
    # Attribute statistics
    _total_attributes = sum(len(n.get("attributes", [])) for n in nodes if isinstance(n, dict))
    _nodes_with_attributes = sum(1 for n in nodes if isinstance(n, dict) and n.get("attributes"))
    _avg_attributes = round(_total_attributes / len(nodes), 2) if nodes else 0.0
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "master_path": str(master_path),
        "nodes": len(nodes),
        "instances": len(instances),
        "relationships": len(relationships),
        "total_attributes": _total_attributes,
        "nodes_with_attributes": _nodes_with_attributes,
        "avg_attributes_per_node": _avg_attributes,
        "layer_distribution": layer_distribution,
        "relationship_layer_distribution": relationship_layer_distribution,
        "relationship_stats": rel_stats,
        "relationship_validation_issues": len(rel_validation_issues),
        "artifacts": artifacts,
        "unknown_governance": unknown_gov_report,
    }
    
    # M2 (CQ_PIPELINE_IMPLEMENTATION_PLAN): CQ review sheet (回译确认) +
    # ArtifactBundle registration. Silent skip when there is no CQ input
    # (零回归约束) — summary_report.json stays byte-identical.
    try:
        from src.pipeline.cq_integration import summarize_cq_review_artifact

        cq_review = summarize_cq_review_artifact(
            getattr(args, "artifact_root", None),
            run_id=str(getattr(args, "run_id", "") or "cq-summary"),
        )
        if cq_review is not None:
            summary["cq_review"] = cq_review
    except Exception:
        pass

    output_dir = Path(args.output_dir)
    json_path = output_dir / "summary_report.json"
    md_path = output_dir / "summary_report.md"

    _write_json(json_path, summary)
    
    # 集中所有的原始Markdown文件和实体文件到artifacts目录
    try:
        artifacts_dir = run_dir / "artifacts" if run_dir else Path(args.output_dir) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. 复制原始Markdown文档
        source_docs_dir = artifacts_dir / "source-documents"
        source_docs_dir.mkdir(exist_ok=True)
        extract_files_dir = run_dir / "extract" / "files" if run_dir else Path(args.output_dir) / "extract" / "files"
        if extract_files_dir.exists():
            for md_file in extract_files_dir.glob("**/*.md"):
                shutil.copy2(md_file, source_docs_dir / md_file.name)
        
        # 2. 复制实体和关系文件
        entities_dir = artifacts_dir / "entities"
        entities_dir.mkdir(exist_ok=True)
        export_dir = run_dir / "export" if run_dir else Path(args.output_dir) / "export"
        if export_dir.exists():
            for entity_file in export_dir.glob("*.json"):
                if "final" in entity_file.name or "entity" in entity_file.name or "relationship" in entity_file.name or "edge" in entity_file.name:
                    shutil.copy2(entity_file, entities_dir / entity_file.name)
        
        # 3. 复制汇总报告和其他重要文件
        shutil.copy2(json_path, artifacts_dir / "summary_report.json")
        if md_path.exists():
            shutil.copy2(md_path, artifacts_dir / "summary_report.md")
        
        # 4. 复制整体manifest文件
        if run_dir:
            manifest_file = run_dir / "manifest.json"
            if manifest_file.exists():
                shutil.copy2(manifest_file, artifacts_dir / "manifest.json")
        
        print(f"[Summarize] ✅ 所有 artifacts 已集中到: {artifacts_dir}", flush=True)
    except Exception as e:
        print(f"[Summarize] ⚠️  集中 artifacts 失败: {e}", flush=True)
    
    md_lines = [
        "# Ontology Pipeline Summary",
        "",
        f"**Generated**: {summary['generated_at']}",
        f"**Master Path**: {summary['master_path']}",
        "",
        "## Entity Count",
        f"- **Nodes**: {summary['nodes']}",
        f"- **Instances**: {summary['instances']}",
        f"- **Relationships**: {summary['relationships']}",
        "",
        "## Attribute Statistics",
        f"- **Total Attributes**: {_total_attributes}",
        f"- **Nodes with Attributes**: {_nodes_with_attributes}/{len(nodes)}",
        f"- **Avg Attributes/Node**: {_avg_attributes}",
        "",
        "## Layer Distribution",
    ]
    
    layer_order = ["L0", "L1", "L2", "L2.5", "L3"]
    for layer in layer_order:
        if layer in layer_distribution:
            md_lines.append(f"- **{layer}**: {layer_distribution[layer]}")
    for layer in sorted(set(layer_distribution.keys()) - set(layer_order)):
        md_lines.append(f"- **{layer}**: {layer_distribution[layer]}")
    
    md_lines.extend([
        "",
        "## Relationship Layer Distribution",
    ])
    if relationship_layer_distribution:
        for layer_key, count in sorted(relationship_layer_distribution.items()):
            md_lines.append(f"- **{layer_key}**: {count}")
    else:
        md_lines.append("- No relationships found")
    
    if rel_stats:
        md_lines.extend([
            "",
            "## Relationship Stats",
            f"- **Total**: {rel_stats.get('total', 0)}",
            f"- **Cross Layer**: {rel_stats.get('cross_layer', 0)}",
            f"- **Same Layer**: {rel_stats.get('same_layer', 0)}",
        ])
    
    md_lines.append(f"- **Validation Issues**: {summary['relationship_validation_issues']}")
    
    if unknown_gov_report and "summary" in unknown_gov_report:
        ug = unknown_gov_report
        md_lines.extend([
            "",
            "## UNKNOWN Governance",
            f"- **Recommendation**: {ug.get('recommendation', 'N/A')}",
            f"- **meta_class UNKNOWN**: {ug['summary'].get('unknown_meta_class_count', 0)} / {ug['summary'].get('total_entities', 0)} ({ug['summary'].get('unknown_meta_class_ratio', 0):.1%})",
            f"- **relation_type UNKNOWN**: {ug['summary'].get('unknown_relation_type_count', 0)} / {ug['summary'].get('total_relations', 0)} ({ug['summary'].get('unknown_relation_type_ratio', 0):.1%})",
        ])
        if ug.get("alerts"):
            md_lines.append("- **Alerts**:")
            for alert in ug["alerts"]:
                md_lines.append(f"  - [{alert['severity']}] {alert['message']}")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    
    return StageResult(
        "summarize",
        True,
        "summarize completed",
        {
            "summary": summary,
            "summary_json_path": str(json_path),
            "summary_md_path": str(md_path),
        },
        round(time.time() - start, 3),
    )