"""Preflight and Health stages."""
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional

from .common import StageResult, PROJECT_ROOT, _read_json, _iter_master_entities, _layer_distribution


def _skill_registry_snapshot(domain: Optional[str] = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "count": 0,
        "skills": [],
        "domain_skills": [],
        "domain": domain or "",
        "error": None,
    }
    try:
        from ontology_compiler.harness import build_default_skill_registry
        registry = build_default_skill_registry()
        skills = registry.list_skills()
        result["available"] = True
        result["count"] = len(skills)
        result["skills"] = [{"name": s.name, "description": s.description} for s in skills]
        if domain:
            result["domain_skills"] = [s.name for s in skills if any(domain.lower() in d.lower() for d in (s.domains or []))]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _discover_schema_assets(root: Optional[Path] = None) -> dict[str, Any]:
    root = root or PROJECT_ROOT
    schemas_dir = root / "config" / "schemas"
    assets: list[dict[str, Any]] = []
    by_kind: dict[str, int] = {}
    
    if schemas_dir.exists():
        for path in sorted(schemas_dir.rglob("*")):
            if not path.is_file():
                continue
            kind = path.suffix.lower().lstrip(".") or "file"
            by_kind[kind] = by_kind.get(kind, 0) + 1
            assets.append({
                "name": path.name,
                "path": str(path.relative_to(root)),
                "abs_path": str(path),
                "kind": kind,
                "size_bytes": path.stat().st_size,
            })
    
    return {
        "available": len(assets) > 0,
        "schema_dir": str(schemas_dir),
        "count": len(assets),
        "assets": assets,
        "by_kind": by_kind,
    }


def _analyze_relationships(relationships: list[dict[str, Any]], entities: list[dict[str, Any]]) -> tuple[list, list, dict]:
    from core.utils.relationship_helpers import (
        collect_entity_ids,
        relationship_endpoints,
    )
    
    def _build_entity_lookup(entities_list):
        lookup = {}
        for e in entities_list:
            eid = str(e.get("meta_id") or e.get("entity_id") or e.get("id") or "").strip()
            if eid:
                lookup[eid] = e
        return lookup
    
    def _normalize_rel_type(rel):
        return str(rel.get("type") or rel.get("relation_type") or rel.get("rel_type") or "").upper().strip()
    
    def _entity_layer_value(n):
        return n.get("layer") or n.get("layer_type") or n.get("canonical_layer") or "L1"
    
    def _validate_oem_relationship(rel_copy, source_entity, target_entity):
        from core.relationship_rules_loader import RelationshipRulesLoader
        rules_loader = RelationshipRulesLoader()
        rules = rules_loader.get_rules()
        
        rel_type = _normalize_rel_type(rel_copy)
        rule = rules.get(rel_type)
        if not rule:
            return rel_copy, None
        
        issue = None
        source_type = str(rel_copy.get("source_type") or "").strip()
        target_type = str(rel_copy.get("target_type") or "").strip()
        
        allowed_source = rule.get("allowed_source_types", [])
        allowed_target = rule.get("allowed_target_types", [])
        family = rule.get("family", "GENERAL")
        rel_copy["relation_family"] = family
        
        if source_type and allowed_source and source_type not in allowed_source:
            issue = {
                "type": "source_type_violation",
                "relationship_type": rel_type,
                "source_type": source_type,
                "allowed_types": allowed_source,
                "severity": str(rule.get("severity", "warn")),
            }
        elif target_type and allowed_target and target_type not in allowed_target:
            issue = {
                "type": "target_type_violation",
                "relationship_type": rel_type,
                "target_type": target_type,
                "allowed_types": allowed_target,
                "severity": str(rule.get("severity", "warn")),
            }
        
        return rel_copy, issue
    
    lookup = _build_entity_lookup(entities)
    enriched: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "total": 0,
        "by_family": {},
        "by_type": {},
        "by_severity": {},
        "cross_layer": 0,
        "same_layer": 0,
        "missing_layer": 0,
        "source_missing": 0,
        "target_missing": 0,
        "layer_matrix": {},
    }
    
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        rel_copy = {**rel}
        source, target = relationship_endpoints(rel_copy)
        source_entity = lookup.get(source)
        target_entity = lookup.get(target)
        
        if source and source_entity is None:
            stats["source_missing"] += 1
        if target and target_entity is None:
            stats["target_missing"] += 1
        
        rel_copy, issue = _validate_oem_relationship(rel_copy, source_entity, target_entity)
        rel_type = _normalize_rel_type(rel_copy) or "RELATED_TO"
        family = str(rel_copy.get("relation_family") or "GENERAL").strip() or "GENERAL"
        stats["by_type"][rel_type] = stats["by_type"].get(rel_type, 0) + 1
        stats["by_family"][family] = stats["by_family"].get(family, 0) + 1
        
        source_layer = str(rel_copy.get("source_layer") or _entity_layer_value(source_entity) or "").strip()
        target_layer = str(rel_copy.get("target_layer") or _entity_layer_value(target_entity) or "").strip()
        
        if source_layer and target_layer:
            if source_layer == target_layer:
                stats["same_layer"] += 1
            else:
                stats["cross_layer"] += 1
            stats["layer_matrix"].setdefault(source_layer, {})
            stats["layer_matrix"][source_layer][target_layer] = stats["layer_matrix"][source_layer].get(target_layer, 0) + 1
        else:
            stats["missing_layer"] += 1
        
        if issue:
            issues.append(issue)
            severity = str(issue.get("severity") or "error").strip() or "error"
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
        
        enriched.append(rel_copy)
    
    stats["total"] = len(enriched)
    return enriched, issues, stats


def _master_template() -> dict[str, Any]:
    return {
        "entities": {
            "L0": [],
            "L1": [],
            "L2": [],
            "L2.5": [],
            "L3": [],
        },
        "nodes": [],
        "instances": [],
        "relationships": [],
        "metadata": {
            "relationship_stats": {},
            "relationship_validation_issues": [],
            "legacy_migrated": False,
        },
        "layer0_metadata": [],
        "layer1_concepts": [],
        "layer2_scenarios": [],
        "layer2_logic": [],
        "layer2_definitions": [],
        "layer25_mapping_rules": [],
    }


def _elapsed(start: float) -> float:
    return round(time.time() - start, 3)


def expand_stages(stage_arg: str, skip_dedup: bool = False) -> list[str]:
    if stage_arg == "full":
        stages = ["preflight", "health", "extract", "merge", "audit", "export", "guard", "summarize"]
        return stages
    return [s.strip() for s in stage_arg.split(",") if s.strip()]


def run_preflight(args: Namespace) -> StageResult:
    start = time.time()
    requested_stages = expand_stages(args.stage, skip_dedup=getattr(args, "skip_dedup", False))
    
    skill_summary = _skill_registry_snapshot(args.domain)
    skills_available = bool(skill_summary.get("available", False) and skill_summary.get("count", 0) > 0)
    skills_count = int(skill_summary.get("count", 0) or 0)
    domain_skills_activated = skill_summary.get("domain_skills", []) if isinstance(skill_summary.get("domain_skills", []), list) else []
    
    llm_connected = False
    llm_error = None
    try:
        # D-1 (2026-08-15): shared client factory (avoids a fresh Unity client
        # per preflight probe and shares circuit-breaker/cache state).
        from src.application.bootstrap.llm_bootstrap import get_llm_client
        client = get_llm_client()
        llm_connected = client is not None
    except Exception as exc:
        llm_connected = False
        llm_error = str(exc)
    
    master_path = Path(args.master_path)
    master_readable = True
    master_exists = master_path.exists()
    if master_exists:
        try:
            if master_path.is_file():
                with master_path.open("rb") as f:
                    f.read(1024)
            elif master_path.is_dir():
                nodes_file = master_path / "nodes.json"
                if nodes_file.exists():
                    with nodes_file.open("rb") as f:
                        f.read(1024)
                else:
                    master_readable = False
        except Exception:
            master_readable = False
    
    source_file_count = 0
    if args.file or args.folder:
        source_path = Path(args.file or args.folder)
        if source_path.exists():
            if source_path.is_file():
                source_file_count = 1
            elif source_path.is_dir():
                try:
                    from ontology_compiler.agents.file_parser import FileParser
                    source_file_count = sum(1 for p in source_path.rglob("*") if p.is_file() and FileParser.can_parse(str(p)))
                except Exception:
                    _SUPPORTED_EXTS = {".md", ".txt", ".pdf", ".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".json"}
                    source_file_count = sum(1 for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS)
    
    master_entity_count = 0
    master_layer_distribution = {"L0": 0, "L1": 0, "L2": 0, "L2.5": 0, "L3": 0, "UNKNOWN": 0}
    master_relationship_stats = {
        "total": 0, "by_family": {}, "by_type": {}, "by_severity": {},
        "cross_layer": 0, "same_layer": 0, "missing_layer": 0,
        "source_missing": 0, "target_missing": 0, "layer_matrix": {},
    }
    master_relationship_issue_count = 0
    preflight_warnings: list[str] = []
    
    if master_exists and master_readable:
        try:
            master_data = _read_json(master_path, _master_template())
            master_entities = _iter_master_entities(master_data, require_meta_id=False)
            master_entity_count = len(master_entities)
            master_layer_distribution = _layer_distribution(master_entities)
            rels = master_data.get("relationships") or master_data.get("edges") or []
            metadata = master_data.get("metadata", {}) if isinstance(master_data.get("metadata", {}), dict) else {}
            rel_stats = metadata.get("relationship_stats", {}) if isinstance(metadata, dict) else {}
            if not isinstance(rel_stats, dict) or not rel_stats:
                _, rel_issues, rel_stats = _analyze_relationships(rels, master_entities)
                master_relationship_issue_count = len(rel_issues)
            else:
                master_relationship_issue_count = int(metadata.get("relationship_validation_issue_count", 0) or 0)
            if not master_relationship_stats["total"]:
                master_relationship_stats = rel_stats if isinstance(rel_stats, dict) else master_relationship_stats
            if master_layer_distribution.get("UNKNOWN", 0) > 0:
                preflight_warnings.append(f"{master_layer_distribution['UNKNOWN']} master entity(ies) have unknown layer")
            if master_relationship_stats.get("missing_layer", 0):
                preflight_warnings.append(f"{master_relationship_stats['missing_layer']} relationship(s) are missing layer metadata")
            if master_relationship_issue_count:
                preflight_warnings.append(f"{master_relationship_issue_count} relationship validation issue(s) already stored in master metadata")
        except Exception as exc:
            preflight_warnings.append(f"failed to inspect master structure: {exc}")
    
    schema_assets = _discover_schema_assets(PROJECT_ROOT)
    checks = {
        "project_root_exists": PROJECT_ROOT.exists(),
        "config_exists": any((PROJECT_ROOT / p).exists() for p in ["config/connections/env.env", "config/env.env", "env.env", "env.local.env"]),
        "pipeline_dir_ready": True,
        "source_selected": ("extract" not in requested_stages) or bool(args.file or args.folder),
        "master_parent_ready": True,
        "skills_available": skills_available,
        "skills_count": skills_count,
        "domain_skills_activated": domain_skills_activated,
        "skill_summary": skill_summary,
        "schema_assets": schema_assets,
        "schema_count": int(schema_assets.get("count", 0) or 0),
        "llm_connected": llm_connected,
        "llm_error": llm_error,
        "master_readable": master_readable,
        "master_exists": master_exists,
        "master_entity_count": master_entity_count,
        "master_layer_distribution": master_layer_distribution,
        "master_relationship_stats": master_relationship_stats,
        "master_relationship_issue_count": master_relationship_issue_count,
        "source_file_count": source_file_count,
        "preflight_warnings": preflight_warnings,
        "preflight_warning_count": len(preflight_warnings),
    }
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.master_path).parent.mkdir(parents=True, exist_ok=True)
    
    thresholds_ok = (
        master_entity_count >= int(getattr(args, "min_master_entities", 1) or 1)
        and int(master_layer_distribution.get("UNKNOWN", 0) or 0) == 0
        and int(master_relationship_issue_count or 0) <= int(getattr(args, "max_relationship_issues", 15) or 15)
        and int(master_relationship_stats.get("missing_layer", 0) or 0) == 0
        and float(master_relationship_stats.get("cross_layer", 0) or 0) <= float(getattr(args, "max_cross_layer_ratio", 0.8) or 0.8)
    )
    checks["thresholds_ok"] = thresholds_ok
    checks["strict_preflight"] = bool(getattr(args, "strict_preflight", False))
    
    success = (
        all(checks[k] for k in ["project_root_exists", "config_exists", "pipeline_dir_ready", "master_readable"])
        and (("extract" not in requested_stages) or checks["source_selected"])
        and (("extract" not in requested_stages) or checks["source_file_count"] > 0)
        and (thresholds_ok or not checks["strict_preflight"])
    )
    
    checks["schema_coverage"] = {
        "available": len(schema_assets.get("assets", [])) > 0,
        "schema_count": int(schema_assets.get("count", 0) or 0),
    }
    checks["skill_coverage"] = {
        "available": skill_summary.get("available", False),
        "skills_count": int(skill_summary.get("count", 0) or 0),
        "domain_skill_count": len(skill_summary.get("domain_skills", [])),
    }

    # M2 (CQ_PIPELINE_IMPLEMENTATION_PLAN): CQ catalog summary + per-run
    # catalog snapshot. Silent skip when no catalog exists so empty-CQ runs
    # stay byte-identical (零回归约束); CQ failures must never break preflight.
    try:
        from src.pipeline.cq_integration import preflight_cq_catalog_artifact, preflight_cq_summary

        cq_summary = preflight_cq_summary()
        if cq_summary is not None:
            checks["cq_loaded_summary"] = cq_summary
            cq_catalog_artifact = preflight_cq_catalog_artifact(getattr(args, "artifact_root", None))
            if cq_catalog_artifact:
                checks["cq_loaded_summary"]["catalog_artifact"] = cq_catalog_artifact
    except Exception:
        pass

    return StageResult("preflight", success, "preflight checks completed", checks, _elapsed(start))


def run_health(args: Namespace) -> StageResult:
    start = time.time()
    data: dict[str, Any] = {}
    try:
        from ontology_compiler.harness import build_default_skill_registry
        registry = build_default_skill_registry()
        data["skills_count"] = len(registry.list_skills())
    except Exception as exc:
        data["skills_error"] = str(exc)
    try:
        from ontology_compiler.services.extraction.unified_compilation_pipeline import UnifiedCompilationService
        data["unified_compilation_service"] = UnifiedCompilationService.__name__
    except Exception as exc:
        data["unified_compilation_service_error"] = str(exc)
    
    master_path = Path(args.master_path)
    data["master_exists"] = master_path.exists()
    schema_assets = _discover_schema_assets(PROJECT_ROOT)
    data["schemas"] = {
        "available": schema_assets.get("count", 0) > 0,
        "schemas_dir": schema_assets.get("schema_dir", ""),
        "schema_count": schema_assets.get("count", 0),
        "schema_files": [asset.get("path", "") for asset in schema_assets.get("assets", [])],
        "schema_names": [asset.get("name", "") for asset in schema_assets.get("assets", [])],
        "json_count": schema_assets.get("by_kind", {}).get("json", 0),
        "yaml_count": schema_assets.get("by_kind", {}).get("yaml", 0) or 0,
    }
    skill_snapshot = _skill_registry_snapshot(getattr(args, "domain", None))
    data["skills"] = {
        "available": skill_snapshot.get("available", False),
        "skills_count": skill_snapshot.get("count", 0),
        "skills": [skill.get("name", "") for skill in skill_snapshot.get("skills", [])],
        "domain_skills": skill_snapshot.get("domain_skills", []),
        "domain": skill_snapshot.get("domain", ""),
        "error": skill_snapshot.get("error", None),
    }
    
    if master_path.exists():
        try:
            master_data = _read_json(master_path, _master_template())
            master_entities = _iter_master_entities(master_data, require_meta_id=False)
            _, rel_issues, rel_stats = _analyze_relationships(
                master_data.get("relationships") or master_data.get("edges") or [],
                master_entities,
            )
            data["master_entity_count"] = len(master_entities)
            data["master_layer_distribution"] = _layer_distribution(master_entities)
            data["master_relationship_stats"] = rel_stats
            data["master_relationship_validation_issue_count"] = len(rel_issues)
            data["master_relationship_validation_issues_sample"] = rel_issues[:10]
        except Exception as exc:
            data["master_inspection_error"] = str(exc)
    
    success = "unified_compilation_service_error" not in data
    return StageResult("health", success, "health checks completed", data, _elapsed(start))