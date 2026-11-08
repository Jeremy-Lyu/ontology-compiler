"""Relationship governance stage."""
import asyncio
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

from .common import StageResult, PROJECT_ROOT, PIPELINE_VALIDATION_THRESHOLDS, _read_json, _iter_master_entities, _entity_layer_value
from src.infrastructure.services.relationship_validity_gate import RelationshipValidityGate
from src.infrastructure.services.relationship_validity_gate import RelationshipValidityGate
from core.utils.relationship_helpers import prune_dangling_relationships


def _ensure_provenance(obj: dict[str, Any], source_file: str, run_id: str, stage: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return obj
    prov = obj.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
    prov.setdefault("source_file", source_file)
    prov.setdefault("run_id", run_id)
    prov.setdefault("stage", stage)
    obj["provenance"] = prov
    obj.setdefault("source_file", source_file)
    obj.setdefault("run_id", run_id)
    obj.setdefault("extraction_stage", stage)
    return obj


def _mark_mbom_signals(entity: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entity, dict):
        return entity
    lname = str(entity.get("name") or entity.get("meta_name") or "").lower()
    if any(token in lname for token in ("occurrence", "position", "variant", "delivery unit", "deliveryunit", "mbom", "bop", "bom")):
        entity.setdefault("mbom_signal", True)
    return entity


async def run_relationship(args: Namespace) -> StageResult:
    start = time.time()
    fragment_artifact = getattr(args, "fragment", None)
    fragment_path = Path(fragment_artifact) if fragment_artifact else None
    
    if not fragment_path or not fragment_path.exists():
        return StageResult("relationship", False, f"Fragment not found: {fragment_path}", elapsed_seconds=round(time.time() - start, 3))
    
    master_path = fragment_path.parent / "master.json"
    if master_path.exists():
        fragment = _read_json(master_path, {})
    else:
        fragment = _read_json(fragment_path, {})
    entities = _iter_master_entities(fragment, require_meta_id=False)
    nodes = [item for item in entities if _entity_layer_value(item) in {"L0", "L1", "L2", "L2.5", "L3"}]
    nodes = [_mark_mbom_signals(_ensure_provenance(n, str(fragment_path), str(getattr(args, "run_id", "")), "relationship")) for n in nodes]
    
    node_map = {str(n.get("meta_id") or n.get("entity_id") or ""): n for n in nodes}
    
    relationships = fragment.get("relationships") or fragment.get("edges") or []
    gate = RelationshipValidityGate()
    gate_result = gate.validate(
        relationships,
        enforce_layer_rules=True,
        enforce_evidence_rules=True,
        min_confidence=PIPELINE_VALIDATION_THRESHOLDS["relationship_min_confidence"],
    )
    valid_relationships = list(gate_result.allowed_relationships)
    violations = [v.to_dict() for v in gate_result.violations]
    
    from ontology_compiler.services.relation_governance import RelationGovernanceEngine
    
    governance = RelationGovernanceEngine()
    critical_missing = {"occurrence": 0, "position": 0, "variant": 0, "provenance": 0}
    
    for rel in valid_relationships:
        if not isinstance(rel, dict):
            continue
            
        original_rel_type = str(rel.get("relation_type") or rel.get("type") or "")
        rel_type = original_rel_type.lower()
        rel["relation_type"] = rel_type
        if "type" in rel:
            rel["type"] = rel_type
        
        source = rel.get("source") or rel.get("from")
        target = rel.get("target") or rel.get("to")
        
        # [FIX] 过滤自引用关系（from/to相同）
        if source == target:
            violations.append({
                "relation_type": rel_type,
                "family": "SELF_REFERENCE",
                "source_layer": "UNKNOWN",
                "target_layer": "UNKNOWN",
                "reason": "self_reference_not_allowed",
            })
            continue

        if isinstance(source, str):
            source_node = node_map.get(source)
            source_layer = _entity_layer_value(source_node) if source_node else "L1"
        elif isinstance(source, dict):
            source_layer = _entity_layer_value(source)
            source_node = source
        else:
            source_layer = "L1"
            source_node = None
        
        if isinstance(target, str):
            target_node = node_map.get(target)
            target_layer = _entity_layer_value(target_node) if target_node else "L1"
        elif isinstance(target, dict):
            target_layer = _entity_layer_value(target)
            target_node = target
        else:
            target_layer = "L1"
            target_node = None

        if source_node:
            sname = str(source_node.get("name") or source_node.get("meta_name") or "").lower()
            if "occurrence" in sname and not (source_node.get("occurrence") or source_node.get("occurrence_id")):
                critical_missing["occurrence"] += 1
            if "position" in sname and not (source_node.get("position") or source_node.get("position_code") or source_node.get("position_label")):
                critical_missing["position"] += 1
            if "variant" in sname and not (source_node.get("variant") or source_node.get("variant_code") or source_node.get("variant_designation")):
                critical_missing["variant"] += 1
            if not (source_node.get("provenance") or source_node.get("source_file") or source_node.get("source_structure") or source_node.get("source_document")):
                critical_missing["provenance"] += 1
        if target_node:
            tname = str(target_node.get("name") or target_node.get("meta_name") or "").lower()
            if "occurrence" in tname and not (target_node.get("occurrence") or target_node.get("occurrence_id")):
                critical_missing["occurrence"] += 1
            if "position" in tname and not (target_node.get("position") or target_node.get("position_code") or target_node.get("position_label")):
                critical_missing["position"] += 1
            if "variant" in tname and not (target_node.get("variant") or target_node.get("variant_code") or target_node.get("variant_designation")):
                critical_missing["variant"] += 1
            if not (target_node.get("provenance") or target_node.get("source_file") or target_node.get("source_structure") or target_node.get("source_document")):
                critical_missing["provenance"] += 1
        
        family = governance.get_family(rel_type)
        rel["relation_family"] = family
    
    fragment["relationships"] = valid_relationships
    fragment["metadata"] = fragment.get("metadata", {})
    fragment["metadata"]["relationship_validation_issues"] = violations
    fragment["metadata"]["critical_missing"] = critical_missing
    
    governed_path = fragment_path.parent / f"governed_{fragment_path.name}"
    with open(governed_path, "w", encoding="utf-8") as f:
        import json
        json.dump(fragment, f, ensure_ascii=False, indent=2)
    
    if any(critical_missing.values()):
        violations.append({
            "relation_type": "CRITICAL_MISSING",
            "family": "MBOM_SAFETY",
            "source_layer": "UNKNOWN",
            "target_layer": "UNKNOWN",
            "reason": f"critical_missing={critical_missing}",
        })

    return StageResult(
        "relationship",
        True,
        f"Relationship governance completed: {len(valid_relationships)} valid, {len(violations)} violations",
        {
            "total_relationships": len(relationships),
            "valid_relationships": len(valid_relationships),
            "violations": violations,
            "violation_count": len(violations),
            "critical_missing": critical_missing,
            "governed_fragment_path": str(governed_path),
            "fragment_path": str(fragment_path),
            "artifact_paths": [str(governed_path), str(fragment_path)],
        },
        round(time.time() - start, 3),
        artifact_paths=[str(governed_path), str(fragment_path)],
        fragment_path=str(fragment_path),
    )
