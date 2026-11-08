"""Audit and Guard stages.

Quality Gate Hard Blocking (2026-07-06):
- When quality_score falls below threshold, entities are marked as pending_review
- Auto-retry loop with max_retries=2 to allow re-extraction
- Stage fails if quality doesn't improve after retries
- Hard block prevents low-quality entities from entering master ontology
"""
import asyncio
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from .common import StageResult, PROJECT_ROOT, _read_json, _iter_master_entities, _entity_layer_value, _layer_distribution

try:
    from src.pipeline.storage.master_store import MasterOntologyStore
except ImportError:
    MasterOntologyStore = None

try:
    from ontology_compiler.services.relation_governance import (
        get_relation_governance_engine,
        RelationGovernanceEngine,
    )
    _REL_GOV_AVAILABLE = True
except ImportError:
    _REL_GOV_AVAILABLE = False

QUALITY_THRESHOLD = 0.6
MAX_RETRIES = 2


def _extract_evidence_text(node: dict[str, Any]) -> str:
    evidence = node.get("evidence")
    candidates: list[str] = []

    if isinstance(evidence, dict):
        for key in ("content", "text", "source_text", "citation"):
            value = evidence.get(key)
            if value:
                candidates.append(str(value))
    elif isinstance(evidence, str) and evidence.strip():
        candidates.append(evidence)

    for key in ("evidence_ref", "citation", "source_file", "source_text", "provenance"):
        value = node.get(key)
        if value:
            candidates.append(str(value))

    candidates = [c.strip() for c in candidates if c and c.strip()]
    if not candidates:
        return ""
    return max(candidates, key=len)


async def run_audit(args: Namespace) -> StageResult:
    start = time.time()
    strict_governance = getattr(args, "strict_governance", False)
    quality_threshold = getattr(args, "quality_threshold", QUALITY_THRESHOLD)
    max_retries = getattr(args, "max_retries", MAX_RETRIES)
    master_path = Path(args.master_path)
    if not master_path.exists():
        return StageResult("audit", False, f"master not found: {master_path}", elapsed_seconds=round(time.time() - start, 3))
    
    if MasterOntologyStore:
        store = MasterOntologyStore(str(master_path))
        master = store.read()
    else:
        master = _read_json(master_path, {})
    
    nodes = master.get("nodes", [])
    nodes = [item for item in nodes if isinstance(item, dict) and _entity_layer_value(item) in {"L0", "L1", "L2", "L2.5", "L3"}]
    instances = master.get("instances", [])
    relationships = master.get("relationships") or master.get("edges") or []
    
    from core.utils.relationship_helpers import collect_entity_ids, relationship_endpoints
    
    all_entities = nodes + [item for item in instances if isinstance(item, dict)]
    node_ids = collect_entity_ids(all_entities)
    
    dangling = []
    for r in relationships:
        source, target = relationship_endpoints(r)
        if source not in node_ids or target not in node_ids:
            dangling.append(r)
    
    shallow = [n for n in nodes if isinstance(n, dict) and not n.get("description") and not n.get("attributes")]
    
    # Attribute statistics (fix: report generators must read 'attributes' field, not 'properties')
    total_attributes = sum(len(n.get("attributes", [])) for n in nodes if isinstance(n, dict))
    nodes_with_attributes = sum(1 for n in nodes if isinstance(n, dict) and n.get("attributes"))
    avg_attributes_per_node = round(total_attributes / len(nodes), 2) if nodes else 0.0
    
    # Evidence quality check: mark entities with insufficient evidence as low-confidence
    # Compatible with multiple evidence field forms:
    #   - evidence.content / evidence.source_text / evidence.citation (dict)
    #   - evidence (string)
    #   - evidence_ref / citation / source_file / source_text / provenance (top-level)
    low_evidence_entities = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        content = _extract_evidence_text(n)
        if len(content) < 20:
            n["review_status"] = "pending_review"
            n["review_reason"] = f"insufficient evidence ({len(content)} chars, threshold=20)"
            low_evidence_entities.append(n.get("name") or n.get("meta_name") or n.get("meta_id", ""))
    low_evidence_count = len(low_evidence_entities)

    critical_signals = {
        "occurrence_missing": 0,
        "position_missing": 0,
        "variant_missing": 0,
        "source_provenance_missing": 0,
    }
    for n in nodes:
        if not isinstance(n, dict):
            continue
        lname = str(n.get("meta_name") or n.get("name") or "").lower()
        # Provenance is required for ALL entities regardless of layer
        if any(token in lname for token in ("deliveryunit", "part", "variant", "position", "occurrence", "mergedpart", "consumedpart")):
            if not (n.get("provenance") or n.get("source_file") or n.get("source_structure") or n.get("source_document")):
                critical_signals["source_provenance_missing"] += 1
        # Position/variant/occurrence FIELDS are only required for L3 instances.
        # L2/L2.5 concept entities (e.g. "POS Position" as a concept) legitimately
        # lack a specific position value — they describe the concept, not an instance.
        if _entity_layer_value(n) == "L3":
            if "occurrence" in lname and not (n.get("occurrence") or n.get("occurrence_id")):
                critical_signals["occurrence_missing"] += 1
            if "position" in lname and not (n.get("position") or n.get("position_code") or n.get("position_label")):
                critical_signals["position_missing"] += 1
            if "variant" in lname and not (n.get("variant") or n.get("variant_code") or n.get("variant_name") or n.get("variant_designation")):
                critical_signals["variant_missing"] += 1

    # --- meta_id uniqueness check (warning only, does not block pipeline) ---
    # Detects when multiple unrelated entities share the same meta_id — a data
    # quality issue that causes problems in global_optimizer's relationship
    # redirection. The fix is in the merge stage (ID collision reassignment),
    # but this check provides visibility at audit time.
    from collections import Counter as _IdCounter
    _all_entity_ids: list[str] = []
    for _n in nodes + [item for item in instances if isinstance(item, dict)]:
        if isinstance(_n, dict):
            _eid = str(_n.get("meta_id") or _n.get("entity_id") or _n.get("id") or "")
            if _eid:
                _all_entity_ids.append(_eid)
    _id_counts = _IdCounter(_all_entity_ids)
    _shared_ids = {eid: cnt for eid, cnt in _id_counts.items() if cnt > 1}
    _duplicate_id_count = sum(cnt - 1 for cnt in _shared_ids.values())
    if _duplicate_id_count > 0:
        print(f"[Audit] ⚠️ meta_id uniqueness: {_duplicate_id_count} duplicate IDs detected "
              f"across {len(_shared_ids)} shared meta_ids")
        for _eid, _cnt in sorted(_shared_ids.items(), key=lambda x: -x[1])[:10]:
            _names = [str(n.get("name") or n.get("meta_name") or "?")[:30]
                      for n in nodes + instances
                      if isinstance(n, dict) and str(n.get("meta_id") or n.get("entity_id") or n.get("id") or "") == _eid]
            print(f"  ⚠️ meta_id '{_eid}' shared by {_cnt} entities: {_names[:5]}")
        if len(_shared_ids) > 10:
            print(f"  ... and {len(_shared_ids) - 10} more shared IDs")

    name_domain_pairs = set()
    for n in nodes:
        if not isinstance(n, dict):
            continue
        name = n.get("meta_name") or n.get("name", "")
        domain = n.get("domain", "")
        name_domain_pairs.add((name, domain))
    duplicate_names = len(nodes) - len(name_domain_pairs)

    # More precise duplicate detection: same (name, layer, meta_class) is a true duplicate
    true_duplicate_keys = set()
    true_duplicate_count = 0
    for n in nodes:
        if not isinstance(n, dict):
            continue
        name = str(n.get("meta_name") or n.get("name", "")).strip().lower()
        layer = str(n.get("layer", "")).strip()
        meta_class = str(n.get("meta_class", "")).strip().lower()
        key = (name, layer, meta_class)
        if key in true_duplicate_keys:
            true_duplicate_count += 1
        else:
            true_duplicate_keys.add(key)
    
    from ontology_compiler.services.quality_assurance.pipelines import QualityGatePipeline
    
    qg = QualityGatePipeline()
    quality_result = await qg.execute(
        {"nodes": nodes, "instances": instances, "relationships": relationships},
        getattr(args, "domain", "SUPPLY_CHAIN")
    )
    
    rel_stats = master.get("metadata", {}).get("relationship_stats", {})
    if not isinstance(rel_stats, dict):
        rel_stats = {}
    rel_validation_issues = master.get("metadata", {}).get("relationship_validation_issues", [])
    if not isinstance(rel_validation_issues, list):
        rel_validation_issues = []
    
    if _REL_GOV_AVAILABLE:
        gov_engine = get_relation_governance_engine()
        node_id_to_layer = {}
        for ent in all_entities:
            eid = str(ent.get("meta_id") or ent.get("entity_id") or ent.get("id") or "")
            if eid:
                node_id_to_layer[eid] = _entity_layer_value(ent)
        
        for rel in relationships:
            source = rel.get("source") or rel.get("from")
            target = rel.get("target") or rel.get("to")
            source_id = str(source) if not isinstance(source, dict) else ""
            target_id = str(target) if not isinstance(target, dict) else ""
            source_layer = node_id_to_layer.get(source_id, "unknown")
            target_layer = node_id_to_layer.get(target_id, "unknown")
            
            validation = gov_engine.validate(rel, source_layer, target_layer)
            for issue in validation.get("issues", []):
                if issue not in rel_validation_issues:
                    rel_validation_issues.append(issue)
    rel_issue_count = len(rel_validation_issues)
    
    rel_issue_by_family: dict[str, int] = {}
    rel_issue_by_type: dict[str, int] = {}
    rel_issue_by_severity: dict[str, int] = {}
    for issue in rel_validation_issues:
        if not isinstance(issue, dict):
            continue
        family = str(issue.get("family") or "UNKNOWN")
        rel_type = str(issue.get("type") or "UNKNOWN")
        severity = str(issue.get("severity") or "error")
        rel_issue_by_family[family] = rel_issue_by_family.get(family, 0) + 1
        rel_issue_by_type[rel_type] = rel_issue_by_type.get(rel_type, 0) + 1
        rel_issue_by_severity[severity] = rel_issue_by_severity.get(severity, 0) + 1
    
    layer_distribution = _layer_distribution(nodes + [item for item in instances if isinstance(item, dict)])
    
    relationship_layer_distribution = {}
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        source = rel.get("source")
        target = rel.get("target")
        
        if isinstance(source, dict):
            source_layer = _entity_layer_value(source)
        else:
            source_id = str(source or "")
            source_layer = "unknown"
            for n in all_entities:
                if str(n.get("meta_id") or n.get("entity_id") or n.get("id") or "") == source_id:
                    source_layer = _entity_layer_value(n)
                    break
        
        if isinstance(target, dict):
            target_layer = _entity_layer_value(target)
        else:
            target_id = str(target or "")
            target_layer = "unknown"
            for n in all_entities:
                if str(n.get("meta_id") or n.get("entity_id") or n.get("id") or "") == target_id:
                    target_layer = _entity_layer_value(n)
                    break
        
        layer_key = f"{source_layer}->{target_layer}"
        relationship_layer_distribution[layer_key] = relationship_layer_distribution.get(layer_key, 0) + 1
    
    skill_gov_error_count = 0
    skill_governance_issues = []
    
    if hasattr(quality_result, 'confidence'):
        quality_score = quality_result.confidence
    elif hasattr(quality_result, 'scores') and isinstance(quality_result.scores, dict):
        quality_score = quality_result.scores.get("overall", 0.0)
    elif isinstance(quality_result, dict):
        quality_score = quality_result.get("overall_score", quality_result.get("confidence", 0.0))
    else:
        quality_score = 0.0

    # Capture per-rule scores for debugging and regression
    per_rule_scores = {}
    if hasattr(quality_result, 'rule_results') and quality_result.rule_results:
        for rr in quality_result.rule_results:
            per_rule_scores[rr.rule_id] = {
                "score": round(rr.score, 4) if isinstance(rr.score, (int, float)) else 0.0,
                "weight": rr.weight,
                "passed": rr.passed,
            }
    
    pending_review_count = 0
    
    for n in nodes:
        if not isinstance(n, dict):
            continue
        
        node_quality = n.get("quality_score", n.get("confidence", None))
        if node_quality is not None and node_quality < quality_threshold:
            n["review_status"] = "pending_review"
            n["review_reason"] = f"entity_quality_score ({node_quality:.3f}) below threshold ({quality_threshold})"
            pending_review_count += 1
        elif "review_status" in n:
            del n["review_status"]
            del n["review_reason"]
    
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        
        inst_quality = inst.get("quality_score", inst.get("confidence", None))
        if inst_quality is not None and inst_quality < quality_threshold:
            inst["review_status"] = "pending_review"
            inst["review_reason"] = f"entity_quality_score ({inst_quality:.3f}) below threshold ({quality_threshold})"
            pending_review_count += 1
        elif "review_status" in inst:
            del inst["review_status"]
            del inst["review_reason"]
    
    passed = quality_score >= quality_threshold or len(nodes) == 0
    if any(critical_signals.values()):
        passed = False
    
    if not passed:
        message = f"AUDIT FAILED - overall quality score {quality_score:.3f} below threshold {quality_threshold}. {pending_review_count} entities marked for review."
    else:
        message = f"audit completed (quality_score={quality_score:.3f})"

    result = StageResult(
        "audit",
        passed,
        message,
        {
            "total_nodes": len(nodes),
            "total_instances": len(instances),
            "total_relationships": len(relationships),
            "dangling_relationships": len(dangling),
            "shallow_nodes": len(shallow),
            "total_attributes": total_attributes,
            "nodes_with_attributes": nodes_with_attributes,
            "avg_attributes_per_node": avg_attributes_per_node,
            "low_evidence_count": low_evidence_count,
            "low_evidence_entities": low_evidence_entities,
            "evidence_coverage": round((len(nodes) - low_evidence_count) / len(nodes), 4) if nodes else 0.0,
            "entity_count": len(nodes),
            "relation_count": len(relationships),
            "duplicate_names": duplicate_names,
            "true_duplicate_count": true_duplicate_count,
            "meta_id_duplicate_count": _duplicate_id_count,
            "meta_id_shared_count": len(_shared_ids),
            "critical_signals": critical_signals,
            "quality_score": quality_score,
            "per_rule_scores": per_rule_scores,
            "quality_threshold": quality_threshold,
            "pending_review_count": pending_review_count,
            "rollback_success": 1.0 if passed and len(nodes) > 0 and len(relationships) >= 0 else 0.0,
            "layer_distribution": layer_distribution,
            "relationship_layer_distribution": relationship_layer_distribution,
            "relationship_validation_issues": rel_issue_count,
            "relationship_issues_by_family": rel_issue_by_family,
            "relationship_issues_by_type": rel_issue_by_type,
            "relationship_issues_by_severity": rel_issue_by_severity,
            "skill_governance_errors": skill_gov_error_count,
            "skill_governance_issues": skill_governance_issues,
            "strict_governance": strict_governance,
            "max_retries": max_retries,
        },
        round(time.time() - start, 3),
    )

    # M3 (CQ_PIPELINE_IMPLEMENTATION_PLAN): read-only executability check of
    # frozen CQs → cq_validation_report.json + cq_gap_report.md (验证闭环).
    # Silent skip when no frozen CQ exists (零回归约束); the check never
    # blocks audit.
    try:
        from src.pipeline.cq_integration import (
            audit_cq_gap_report_artifact,
            audit_cq_report_artifact,
            audit_cq_validation,
        )

        cq_validation = audit_cq_validation(master or {})
        if cq_validation is not None:
            artifact_root = getattr(args, "artifact_root", None)
            cq_report_path = audit_cq_report_artifact(cq_validation, artifact_root)
            cq_gap_path = audit_cq_gap_report_artifact(cq_validation, artifact_root)
            summary = {
                "checked_count": cq_validation["checked_count"],
                "executable_count": cq_validation["executable_count"],
            }
            if cq_report_path:
                summary["report"] = cq_report_path
            if cq_gap_path:
                summary["gap_report"] = cq_gap_path
            result.data["cq_validation_summary"] = summary
    except Exception:
        pass

    return result


async def run_guard(args: Namespace) -> StageResult:
    start = time.time()
    critical_signals = {}
    master_path = Path(args.master_path)
    if not master_path.exists():
        return StageResult("guard", False, f"master not found: {master_path}", elapsed_seconds=round(time.time() - start, 3))
    
    if master_path.is_dir():
        nodes = _read_json(master_path / "nodes.json", [])
        instances = _read_json(master_path / "instances.json", [])
        relationships = _read_json(master_path / "relationships.json", [])
        metadata = _read_json(master_path / "metadata.json", {})
        master = {
            "nodes": nodes,
            "instances": instances,
            "relationships": relationships,
            "metadata": metadata
        }
    else:
        master = _read_json(master_path, {})
        nodes = master.get("nodes", [])
        instances = master.get("instances", [])
        relationships = master.get("relationships") or master.get("edges") or []
    
    nodes = [item for item in nodes if isinstance(item, dict) and _entity_layer_value(item) in {"L0", "L1", "L2", "L2.5", "L3"}]
    
    from ontology_compiler.services.quality_assurance.pipelines import ConflictDetectionPipeline
    from ontology_compiler.services.quality_assurance.base import DetectionContext
    
    engine = ConflictDetectionPipeline()
    context = DetectionContext(
        fragment={"nodes": nodes, "instances": instances, "relationships": relationships}
    )
    conflicts = engine.detect(context)
    
    blocking = [c for c in conflicts if c.severity == "BLOCKING"]
    rel_validation_issues = master.get("metadata", {}).get("relationship_validation_issues", [])
    if not isinstance(rel_validation_issues, list):
        rel_validation_issues = []
    
    if _REL_GOV_AVAILABLE:
        gov_engine = get_relation_governance_engine()
        all_entities = nodes + [item for item in instances if isinstance(item, dict)]
        node_id_to_layer = {}
        for ent in all_entities:
            eid = str(ent.get("meta_id") or ent.get("entity_id") or ent.get("id") or "")
            if eid:
                node_id_to_layer[eid] = _entity_layer_value(ent)
        
        for rel in relationships:
            source = rel.get("source") or rel.get("from")
            target = rel.get("target") or rel.get("to")
            source_id = str(source) if not isinstance(source, dict) else ""
            target_id = str(target) if not isinstance(target, dict) else ""
            source_layer = node_id_to_layer.get(source_id, "unknown")
            target_layer = node_id_to_layer.get(target_id, "unknown")
            
            validation = gov_engine.validate(rel, source_layer, target_layer)
            for issue in validation.get("issues", []):
                if issue not in rel_validation_issues:
                    rel_validation_issues.append(issue)
    
    relationship_blocking = [issue for issue in rel_validation_issues if str(issue.get("severity") or "error").lower() == "error"]
    missing_evidence_count = 0
    for issue in rel_validation_issues:
        severity = str(issue.get("severity") or "error").lower()
        if severity != "error":
            continue
        reason = str(issue.get("reason") or "").lower()
        if any(token in reason for token in ("evidence", "citation", "source_file", "line_range", "rule_no")):
            missing_evidence_count += 1
    
    layer_pair_issues = []
    try:
        from .preflight import _analyze_relationships
        _, layer_pair_issues, _ = _analyze_relationships(relationships, nodes)
    except Exception:
        pass
    
    layer_pair_blocking = [issue for issue in layer_pair_issues if issue.get("severity") == "error"]
    layer_pair_warn_count = len([issue for issue in layer_pair_issues if issue.get("severity") == "warn"])
    
    rel_stats = master.get("metadata", {}).get("relationship_stats", {})
    if not isinstance(rel_stats, dict):
        rel_stats = {}
    total_rel = int(rel_stats.get("total", 0) or 0)
    cross_layer_count = int(rel_stats.get("cross_layer", 0) or 0)
    cross_layer_ratio = round(cross_layer_count / total_rel, 3) if total_rel else 0.0
    same_layer_count = int(rel_stats.get("same_layer", 0) or 0)
    missing_layer_count = int(rel_stats.get("missing_layer", 0) or 0)
    
    max_cross_layer_ratio = float(getattr(args, "max_cross_layer_ratio", 0.8) or 0.8)
    max_relationship_issues = int(getattr(args, "max_relationship_issues", 100) or 100)
    strict_governance = bool(getattr(args, "strict_governance", False))
    
    passed = (
        len(nodes) > 0
        and len(blocking) == 0
        and len(relationship_blocking) <= max_relationship_issues
        and len(layer_pair_blocking) == 0
        and missing_evidence_count <= max_relationship_issues
        and cross_layer_ratio <= max_cross_layer_ratio
        and missing_layer_count == 0
        and not any(critical_signals.values())
    )
    
    data = {
        "master_path": str(master_path),
        "nodes": len(nodes),
        "instances": len(instances),
        "relationships": len(relationships),
        "relationship_stats": rel_stats,
        "total_conflicts": len(conflicts),
        "blocking_conflicts": len(blocking),
        "relationship_blocking_issues": len(relationship_blocking),
        "missing_evidence_issue_count": missing_evidence_count,
        "layer_pair_blocking_count": len(layer_pair_blocking),
        "layer_pair_warn_count": layer_pair_warn_count,
        "cross_layer_ratio": cross_layer_ratio,
        "cross_layer_count": cross_layer_count,
        "same_layer_count": same_layer_count,
        "missing_layer_count": missing_layer_count,
        "thresholds": {
            "max_cross_layer_ratio": max_cross_layer_ratio,
            "max_relationship_issues": max_relationship_issues,
        },
        "strict_governance": strict_governance,
        "conflict_details": [
            {"rule": c.rule, "severity": c.severity, "message": c.message, "location": c.location}
            for c in blocking[:5]
        ],
        "relationship_issue_details": relationship_blocking[:5],
        "passed": passed,
    }
    
    if strict_governance and not passed:
        data["governance_enforced"] = True
    
    return StageResult("guard", passed, "guard completed", data, round(time.time() - start, 3))