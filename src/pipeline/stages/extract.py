"""Extract stage."""
import asyncio
import json
import logging
import os
import time
from argparse import Namespace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from .common import StageResult, _read_json, _write_json, _infer_domain_from_files, fragment_has_content, infer_document_type
from ontology_compiler.agents.file_parser import FileParser
from ..run_context import build_pipeline_run_paths

logger = logging.getLogger("ontology.extract")
logger.setLevel(logging.INFO)

# Add a stream handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

def _file_timeout_seconds() -> int:
    """Per-file extract ceiling (seconds), configurable via env.

    DeepSeek calls routinely take 3-6 min per large chunk; with smart chunking
    (up to N chunks at concurrency) the whole file can legitimately need >10
    min. The hardcoded 600s ceiling caused VSM/ATACQ runs to die mid-extract
    when a slow chunk + JSON retries accumulated. Default 300s; raise via
    ``EXTRACT_FILE_TIMEOUT``.
    """
    value = os.environ.get("EXTRACT_FILE_TIMEOUT", "300")
    try:
        return max(300, int(value))
    except ValueError:
        return 300

def _file_concurrency() -> int:
    """File-level extraction concurrency, configurable via env.

    Default 8. DIFY/DeepSeek API rate limiting is owned by the LLM client
    (RateLimiter); if 429s appear, lower this or raise
    DIFY_RATE_LIMIT_MAX_CALLS / DIFY_RATE_LIMIT_WINDOW_S.
    """
    value = os.environ.get("EXTRACT_FILE_CONCURRENCY", "8")
    try:
        return max(1, int(value))
    except ValueError:
        return 8

# Phase 3 LOC reduction (docs/CODE_CLEANUP_CANDIDATES.md §3.1): the cross-file
# [GLOBAL ID FIX] cluster moved verbatim to global_id_fix.py; re-imported so
# internal call sites and `extract._apply_global_id_uniqueness`-style access
# keep working (behavior locked by tests/pipeline/test_extract_characterization.py).
from .global_id_fix import (  # noqa: E402,F401  (re-exported)
    _build_global_id_indexes,
    _remap_colliding_node_ids,
    _rewire_colliding_relationships,
    _report_global_id_uniqueness,
    _apply_global_id_uniqueness,
)

from .domain_safety_net import (
    append_candidate_entities,
    append_candidate_relationships,
    append_deterministic_domain_relations,
    ensure_deterministic_domain_entities,
)

try:
    from src.infrastructure.services.document_cache import DocumentCache
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False

try:
    from src.infrastructure.llm.llm_candidate_generator import LLMCandidateGenerator
    _LLM_CANDIDATE_GENERATOR_AVAILABLE = True
except ImportError:
    _LLM_CANDIDATE_GENERATOR_AVAILABLE = False

try:
    from ontology_compiler.services.l25_skill_loader import get_l25_skill_loader
    _L25_SKILLS_AVAILABLE = True
except ImportError:
    _L25_SKILLS_AVAILABLE = False

try:
    from src.infrastructure.config.unified_config import get_unified_config, get_unified_config_diagnostics
    _UNIFIED_CONFIG_AVAILABLE = True
except ImportError:
    _UNIFIED_CONFIG_AVAILABLE = False

try:
    from ontology_compiler.services.relation_governance import get_relation_governance_engine
    _RELATION_GOVERNANCE_AVAILABLE = True
except ImportError:
    _RELATION_GOVERNANCE_AVAILABLE = False

def _log_unified_config_diagnostics() -> None:
    if not _UNIFIED_CONFIG_AVAILABLE:
        return
    try:
        d = get_unified_config_diagnostics()
        v = d.get("validation") or {}
        e = d.get("entity_classification") or {}
        r = d.get("relationship_rules") or {}
        print(f"[Extract][CONFIG] status={d.get('status', 'unknown')} validation={v.get('status', 'unknown')} entity_rules={e.get('rule_count', 0)} relation_rules={r.get('relation_rules', 0)} issues={v.get('issue_count', 0)}", flush=True)
    except Exception as exc:
        print(f"[Extract][CONFIG] diagnostics unavailable: {exc}", flush=True)

def _ensure_provenance(entity: dict[str, Any], source_file: str, run_id: str, stage: str) -> dict[str, Any]:
    if not isinstance(entity, dict):
        return entity
    prov = entity.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
    prov.setdefault("source_file", source_file)
    prov.setdefault("run_id", run_id)
    prov.setdefault("stage", stage)
    entity["provenance"] = prov
    entity.setdefault("source_file", source_file)
    entity.setdefault("run_id", run_id)
    entity.setdefault("extraction_stage", stage)

    evidence = entity.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {} if evidence is None else {"content": str(evidence)}
    if not evidence.get("content"):
        evidence["content"] = (
            str(entity.get("source_text") or "")
            or str(entity.get("citation") or "")
            or str(entity.get("evidence_ref") or "")
            or f"Source: {source_file}"
        )
    if not evidence.get("source_file"):
        evidence["source_file"] = source_file
    if "confidence" not in evidence:
        evidence["confidence"] = float(entity.get("confidence", 0.7) or 0.7)
    entity["evidence"] = evidence
    return entity

_SYNONYM_NORMALIZATION_MAP = {
    "saps4": "sap s4",
    "sap s/4": "sap s4",
    "sap s-4": "sap s4",
    "mbomand bop": "mbom and bop",
    "manage mbomand bop": "manage mbom and bop",
    "deliveryunit": "delivery unit",
    "delivery unit bom": "delivery unit",
    "position in bom": "position",
    "position pos1": "position pos1",
    "position pos2": "position pos2",
    "ebom_structure": "ebom structure",
    "mbom_structure": "mbom structure",
    "bop_process": "bop process",
    "pos_physicalposition": "position physical position",
    "pii_knowledge": "pii knowledge",
    "cnote_entity": "change note entity",
}

_ALIASES_CACHE_KEY = "_alias_lookup_cache"

@lru_cache(maxsize=1)
def _load_entity_aliases() -> dict[str, Any]:
    alias_path = _config_path("config/entity_aliases.yaml")
    if not alias_path.exists() or yaml is None:
        return {}
    try:
        with alias_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _build_alias_lookup() -> dict[str, dict[str, str]]:
    raw = _load_entity_aliases()
    aliases = raw.get("aliases") or []
    lookup: dict[str, dict[str, str]] = {}
    for item in aliases:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or "").strip()
        meta_class = str(item.get("meta_class") or "").strip()
        if not canonical:
            continue
        for variant in [canonical, *(item.get("variants") or [])]:
            if not isinstance(variant, str):
                continue
            key = _normalize_semantic_name(variant)
            if not key:
                continue
            lookup[key] = {"canonical": canonical, "meta_class": meta_class}
    return lookup

@lru_cache(maxsize=1)
def _alias_lookup() -> dict[str, dict[str, str]]:
    return _build_alias_lookup()

def _normalize_semantic_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return name
    n = name.strip().lower().replace("_", " ").replace("-", " ")
    n = " ".join(n.split())
    if n in _SYNONYM_NORMALIZATION_MAP:
        return _SYNONYM_NORMALIZATION_MAP[n]
    return n

def _mark_mbom_signals(entity: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entity, dict):
        return entity
    lname = _normalize_semantic_name(str(entity.get("name") or entity.get("meta_name") or entity.get("description") or ""))
    if any(token in lname for token in ("occurrence", "position", "variant", "delivery unit", "mbom", "bop", "bom", "alignment", "dedup", "change note")):
        entity.setdefault("mbom_signal", True)
    return entity

def _normalize_entity_payload(entity: dict[str, Any], source_file: str, run_id: str, stage: str = "extract") -> dict[str, Any]:
    entity = _ensure_provenance(entity, source_file, run_id, stage)
    if entity.get("name"):
        entity["name"] = _normalize_semantic_name(entity["name"])
    if entity.get("meta_name"):
        entity["meta_name"] = _normalize_semantic_name(entity["meta_name"])
    if entity.get("description"):
        entity["description"] = _normalize_semantic_name(entity["description"])
    entity = _mark_mbom_signals(entity)
    entity.setdefault("decision_trace", []).append({
        "action": "normalize_text_fields",
        "reason": f"semantic normalization during {stage}",
        "rule_source": f"extract.{stage}.normalization",
    })
    return entity

def _normalize_entity_collection(items: list[Any], source_file: str, run_id: str, stage: str = "extract") -> list[Any]:
    normalized_items = []
    for item in items:
        if isinstance(item, dict):
            item = _normalize_entity_payload(item, source_file, run_id, stage)
        normalized_items.append(item)
    return normalized_items

def _assign_missing_entity_ids(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign layer-sequenced meta_ids to entities lacking one (e.g. safety-net
    backfilled canonical entities), so downstream merge can resolve endpoints."""
    layer_max: dict[str, int] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        eid = str(n.get("meta_id") or n.get("entity_id") or n.get("id") or "")
        if eid and "-" in eid:
            prefix, _, num_str = eid.rpartition("-")
            try:
                layer_max[prefix] = max(layer_max.get(prefix, 0), int(num_str))
            except ValueError:
                pass
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if n.get("meta_id") or n.get("entity_id") or n.get("id"):
            continue
        layer = str(n.get("layer") or "L2")
        layer_max[layer] = layer_max.get(layer, 0) + 1
        n["meta_id"] = f"{layer}-{layer_max[layer]:03d}"
        n.setdefault("decision_trace", []).append({
            "action": "assign_missing_entity_id",
            "reason": "backfilled canonical entity needs an id for merge resolution",
            "rule_source": "extract.run_extract.id_assignment",
        })
    return nodes

def _normalize_relationship_collection(items: list[Any], source_file: str, run_id: str, stage: str = "extract") -> list[Any]:
    normalized_items = []
    for item in items:
        if isinstance(item, dict):
            prov = item.get("provenance")
            if not isinstance(prov, dict):
                prov = {}
            prov.setdefault("source_file", source_file)
            prov.setdefault("run_id", run_id)
            prov.setdefault("stage", stage)
            item["provenance"] = prov
            item.setdefault("decision_trace", []).append({
                "action": "normalize_relationship_provenance",
                "reason": "attach provenance before output",
                "rule_source": f"extract.{stage}.relationship_normalization",
            })
        normalized_items.append(item)
    return normalized_items

def _apply_relationship_governance(items: list[Any], domain: str) -> list[Any]:
    if not _RELATION_GOVERNANCE_AVAILABLE:
        return items

    try:
        engine = get_relation_governance_engine()
    except Exception:
        return items

    governed: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            governed.append(item)
            continue

        validation = engine.validate(item)
        relation_type = validation.get("relation_type") or engine._read_relation_type(item)
        family = validation.get("family") or engine.get_family(relation_type)
        severity = validation.get("severity") or engine.relation_severity(relation_type)
        result_class = str(validation.get("result_class") or "valid").lower()
        rewrite_target = validation.get("rewrite_target")
        rewrite_rank = validation.get("rewrite_rank")
        fallback_reason = validation.get("fallback_reason")
        classification = "valid"
        if result_class == "rewrite":
            classification = "rewrite"
        elif result_class == "fallback":
            classification = "fallback"

        item.setdefault("relation_family", family)
        item.setdefault("relation_severity", severity)
        item.setdefault("relation_validation", validation)
        item["relation_result_class"] = classification

        if classification == "rewrite":
            normalized = engine.normalize(item)
            item.update({k: v for k, v in normalized.items() if k in {"relation_type", "rewritten_relation_type", "rewrite_reason", "relation_family", "decision_trace", "rewrite_rank"}})
            item["rewrite_reason"] = item.get("rewrite_reason") or fallback_reason or f"governance rewrite for family={family}"
            if rewrite_target and not item.get("rewritten_relation_type"):
                item["rewritten_relation_type"] = rewrite_target
            if rewrite_rank is not None:
                item["rewrite_rank"] = rewrite_rank
            item.setdefault("decision_trace", []).append({
                "action": "rewrite_relationship",
                "reason": fallback_reason or "relationship governance rewrite",
                "before": relation_type,
                "after": rewrite_target or relation_type,
                "rewrite_rank": rewrite_rank,
                "rule_source": "extract.relationship_governance",
            })
        elif classification == "fallback":
            item["fallback_reason"] = fallback_reason or "relationship governance fallback"
            if not relation_type:
                item["relation_type"] = "RELATED_TO"
            item["rewrite_reason"] = fallback_reason or "governance fallback retained"
            item.setdefault("decision_trace", []).append({
                "action": "fallback_relationship",
                "reason": fallback_reason or "relationship governance fallback",
                "before": relation_type,
                "after": item.get("relation_type") or "RELATED_TO",
                "rule_source": "extract.relationship_governance",
            })

        item.setdefault("governance_result", {
            "class": classification,
            "family": family,
            "relation_type": relation_type,
            "rewrite_target": rewrite_target,
            "rewrite_rank": rewrite_rank,
            "fallback_reason": fallback_reason,
        })
        governed.append(item)
    return governed

def _annotate_fragment_entities(fragment_data: dict[str, Any], source_file: str, run_id: str) -> dict[str, Any]:
    if not isinstance(fragment_data, dict):
        return fragment_data
    for bucket_key in ("nodes", "instances"):
        bucket = fragment_data.get(bucket_key)
        if isinstance(bucket, list):
            for idx, item in enumerate(bucket):
                if isinstance(item, dict):
                    bucket[idx] = _normalize_entity_payload(item, source_file, run_id, "extract")
    rels = fragment_data.get("relationships")
    if isinstance(rels, list):
        for idx, rel in enumerate(rels):
            if isinstance(rel, dict):
                rel_prov = rel.get("provenance")
                if not isinstance(rel_prov, dict):
                    rel_prov = {}
                rel_prov.setdefault("source_file", source_file)
                rel_prov.setdefault("run_id", run_id)
                rel_prov.setdefault("stage", "extract")
                rel["provenance"] = rel_prov
    return fragment_data

def _normalize_l0_entities(nodes: list[Any], rules: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rules = rules or _load_l0_classification_rules()
    valid_l0_classes = set(rules.get("valid_l0_classes") or ["Capability", "System", "Actor", "Asset", "Data", "Process", "Concept"])
    prohibited_examples = rules.get("l0_prohibited_examples") or []
    prohibited_lookup = {str(item.get("name", "")).strip().lower() for item in prohibited_examples if isinstance(item, dict)}

    l0_nodes = [n for n in nodes if isinstance(n, dict) and n.get("layer") == "L0"]
    non_l0_nodes = [n for n in nodes if isinstance(n, dict) and n.get("layer") != "L0"]

    filtered_l0 = []
    valid_l0_lower = {c.lower() for c in valid_l0_classes}
    for n in l0_nodes:
        name = n.get("name", "").strip()
        meta_class = n.get("meta_class", "").strip()
        if name.lower() in prohibited_lookup:
            print(f"[Extract] L0 filter: prohibited example '{name}' moved to L1")
            n["layer"] = "L1"
            n["parent_class"] = meta_class or name
            non_l0_nodes.append(n)
            continue
        if name.lower() in valid_l0_lower or meta_class.lower() in valid_l0_lower:
            for valid_class in valid_l0_classes:
                if name.lower() == valid_class.lower():
                    n["name"] = valid_class
                    n["meta_class"] = valid_class
                    break
            filtered_l0.append(n)
        else:
            print(f"[Extract] L0 filtering: moving '{name}' (meta_class={meta_class}) to L1 as domain-specific concept")
            n["layer"] = "L1"
            n["parent_class"] = meta_class
            non_l0_nodes.append(n)

    seen_l0_keys = set()
    deduped_l0 = []
    for n in filtered_l0:
        key = f"{n.get('name', '').lower()}_{n.get('meta_class', '').lower()}"
        if key not in seen_l0_keys:
            seen_l0_keys.add(key)
            deduped_l0.append(n)
    if len(deduped_l0) < len(filtered_l0):
        print(f"[Extract] L0 dedup: {len(filtered_l0)} -> {len(deduped_l0)} (removed {len(filtered_l0) - len(deduped_l0)} duplicates)")

    if not deduped_l0:
        print(f"[Extract] L0 layer is empty, auto-supplementing base classes")
        base_l0_classes = list(valid_l0_classes)
        for i, cls in enumerate(base_l0_classes, 1):
            deduped_l0.append({
                "meta_id": f"L0-{i:03d}",
                "name": cls,
                "meta_class": cls,
                "layer": "L0",
                "description": f"Abstract {cls} category (auto-supplemented)",
                "evidence": {"source_text": f"Auto-supplemented L0 {cls}", "confidence": 0.9},
                "attributes": [{"name": "auto_generated", "value": "true"}],
            })
        print(f"[Extract] Added {len(base_l0_classes)} L0 base classes")

    return deduped_l0, non_l0_nodes

def _apply_l25_rule_safety_net(all_nodes: list[Any]) -> None:
    rule_types_to_map = {"businessrule", "validationrule", "mappingrule", "business_rule", "validation_rule", "mapping_rule"}
    for node in all_nodes:
        if isinstance(node, dict) and node.get("layer") == "L2.5":
            meta_class = node.get("meta_class", "").lower()
            if any(rule_type in meta_class for rule_type in rule_types_to_map):
                print(f"[Extract] L2.5 safety net: mapping '{node.get('name')}' (meta_class={node.get('meta_class')}) to Data")
                node["meta_class"] = "Data"
                node["meta_subclass"] = node.get("meta_subclass") or "Rule_Logic"
                decision_trace = node.setdefault("decision_trace", [])
                if isinstance(decision_trace, list):
                    decision_trace.append({
                        "action": "safety_net_map_to_data",
                        "reason": "L2.5 rule types normalized to Data",
                        "before": meta_class,
                        "after": "Data",
                        "rule_source": "extract.run_extract.safety_net",
                    })

def _apply_meta_class_safety_net(all_nodes: list[Any]) -> None:
    actor_kws = ["engineer", "buyer", "manager", "analyst", "specialist", "coordinator", "planner", "operator", "director", "officer", "consultant", "supervisor"]
    data_kws = ["time", "window", "date", "period", "duration", "value", "calculation", "fluctuation", "quantity", "amount", "rate", "threshold", "lead time"]
    system_kws = ["calculator", "converter", "portal", "platform", "tool", "application", "database", "registry", "tracker", "dashboard", "module"]
    for node in all_nodes:
        if not isinstance(node, dict):
            continue
        name = node.get("name", "").lower().strip()
        mc = node.get("meta_class", "")
        if (name.endswith("department") or name.endswith("team") or name.endswith("organization") or any(kw in name for kw in actor_kws)):
            if mc != "Actor":
                print(f"[Extract] meta_class fix: '{node.get('name','')}' {mc} -> Actor")
                node["meta_class"] = "Actor"
                node.setdefault("decision_trace", []).append({
                    "action": "normalize_meta_class",
                    "reason": "actor keyword heuristic",
                    "before": mc,
                    "after": "Actor",
                    "rule_source": "extract.run_extract.meta_class_fix",
                })
        elif any(kw in name for kw in data_kws):
            if mc in ("System", "Asset", "Actor"):
                print(f"[Extract] meta_class fix: '{node.get('name','')}' {mc} -> Data")
                node["meta_class"] = "Data"
                node.setdefault("decision_trace", []).append({
                    "action": "normalize_meta_class",
                    "reason": "data keyword heuristic",
                    "before": mc,
                    "after": "Data",
                    "rule_source": "extract.run_extract.meta_class_fix",
                })
        elif any(kw in name for kw in system_kws):
            if mc not in ("System",):
                print(f"[Extract] meta_class fix: '{node.get('name','')}' {mc} -> System")
                node["meta_class"] = "System"
                node.setdefault("decision_trace", []).append({
                    "action": "normalize_meta_class",
                    "reason": "system keyword heuristic",
                    "before": mc,
                    "after": "System",
                    "rule_source": "extract.run_extract.meta_class_fix",
                })

_LAYER_PROMOTIONS = {
    "supply chain": "L1",
    "organization role": "L1",
    "config manager": "L1",
    "accountability check": "L1",
}

_LAYER_DOWNGRADES = {
    "ebom release": "L2.5",
    "mbom release": "L2.5",
    "bop process": "L2.5",
}

_NAME_REWRITES = {
    "co specialist": "CO Specialist",
    "tc plm": "Teamcenter PLM",
    "art": "ART (Anti-Rattle Tab)",
    "fg": "FG (Finished Goods)",
    "pos": "POS (Position)",
    "bev": "BEV (Battery Electric Vehicle)",
    "phev": "PHEV (Plug-in Hybrid EV)",
    "bop to prins investigation": "BOP-to-PRINS Investigation",
    "cnote entity": "Change Note Entity",
    "pos physicalposition": "Position Physical Position",
    "pii knowledge": "PII Knowledge",
}

_ATTR_OVERRIDES = {
    "caliper": ["part_number", "material_type"],
    "fluid": ["part_number", "material_type"],
    "caliper cover": ["part_number", "material_type"],
    "rear seat": ["part_number", "material_type"],
    "protective foil": ["part_number", "material_type"],
    "ebom policy": ["rule_id", "rule_text", "enforcement"],
    "mbom policy": ["rule_id", "rule_text", "enforcement"],
}

_MERGE_INTO_PARENT = {
    "ebom structure": "ebom",
    "mbom structure": "mbom",
}

def _config_path(relative_path: str) -> Path:
    return Path(__file__).resolve().parents[3] / relative_path

@lru_cache(maxsize=1)
def _load_l0_classification_rules() -> dict[str, Any]:
    if _UNIFIED_CONFIG_AVAILABLE:
        try:
            cfg = get_unified_config()
            if getattr(cfg, "l0_classification_rules", None):
                return cfg.l0_classification_rules
        except Exception:
            pass
    rules_path = _config_path("config/ontology/l0_classification_rules.yaml")
    if not rules_path.exists() or yaml is None:
        return {}
    try:
        with rules_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

@lru_cache(maxsize=1)
def _load_entity_classification_overlay() -> dict[str, Any]:
    if _UNIFIED_CONFIG_AVAILABLE:
        try:
            cfg = get_unified_config()
            overlay = getattr(cfg, "entity_classification_unified", None)
            if overlay:
                return overlay
        except Exception:
            pass
    rules_path = _config_path("config/ontology/entity_classification_unified.yaml")
    if not rules_path.exists() or yaml is None:
        return {}
    try:
        with rules_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _merge_classification_maps(base_map: dict[str, Any], overlay_map: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_map or {})
    for key, value in (overlay_map or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged

def _load_optimization_config() -> dict[str, dict]:
    """Load classification overlay and extend the optimization maps from it.

    Returns merged maps: layer_promotions, layer_downgrades, name_rewrites,
    attr_overrides, merge_into_parent.
    """
    overlay = _load_entity_classification_overlay()
    overlay_layers = overlay.get("layers") or {}
    overlay_types = overlay.get("types") or {}
    overlay_rules = overlay.get("classification_rules") or []
    node_optimization_config = overlay.get("node_optimizations") or {}

    layer_promotions = _merge_classification_maps(_LAYER_PROMOTIONS, node_optimization_config.get("layer_promotions"))
    layer_downgrades = _merge_classification_maps(_LAYER_DOWNGRADES, node_optimization_config.get("layer_downgrades"))
    name_rewrites = _merge_classification_maps(_NAME_REWRITES, node_optimization_config.get("name_rewrites"))
    attr_overrides = _merge_classification_maps(_ATTR_OVERRIDES, node_optimization_config.get("attr_overrides"))
    merge_into_parent = _merge_classification_maps(_MERGE_INTO_PARENT, node_optimization_config.get("merge_into_parent"))

    # Overlay may extend defaults, but it should not weaken the current behavior.
    for type_name, type_spec in overlay_types.items():
        if not isinstance(type_spec, dict):
            continue
        target_layer = str(type_spec.get("layer") or "").upper().strip()
        normalized_name = _normalize_semantic_name(str(type_name))
        if target_layer == "L1" and normalized_name:
            layer_promotions.setdefault(normalized_name, "L1")
        elif target_layer == "L2.5" and normalized_name:
            layer_downgrades.setdefault(normalized_name, "L2.5")

    for rule in overlay_rules:
        if not isinstance(rule, dict):
            continue
        result = rule.get("result") or {}
        result_type = _normalize_semantic_name(str(result.get("type") or ""))
        result_layer = str(result.get("layer") or "").upper().strip()
        if result_type and result_layer == "L1":
            layer_promotions.setdefault(result_type, "L1")
        elif result_type and result_layer == "L2.5":
            layer_downgrades.setdefault(result_type, "L2.5")

    for layer_name, layer_spec in overlay_layers.items():
        if not isinstance(layer_spec, dict):
            continue
        if str(layer_name).upper() == "L2.5":
            for type_name in layer_spec.get("types") or []:
                normalized_name = _normalize_semantic_name(str(type_name))
                if normalized_name:
                    layer_downgrades.setdefault(normalized_name, "L2.5")

    return {
        "layer_promotions": layer_promotions,
        "layer_downgrades": layer_downgrades,
        "name_rewrites": name_rewrites,
        "attr_overrides": attr_overrides,
        "merge_into_parent": merge_into_parent,
    }

def _optimize_node(
    node: dict[str, Any],
    *,
    layer_promotions: dict,
    layer_downgrades: dict,
    name_rewrites: dict,
    attr_overrides: dict,
    merge_into_parent: dict,
) -> dict[str, Any] | None:
    """Apply optimized classification to a single node in place.

    Returns the node, or None when it was merged into a parent (dropped).
    """
    name = _normalize_semantic_name(str(node.get("name") or node.get("meta_name") or ""))
    meta_class = str(node.get("meta_class") or "")
    decision_trace = list(node.get("decision_trace") or []) if isinstance(node.get("decision_trace"), list) else []

    def record_decision(action: str, reason: str, before: Any = None, after: Any = None) -> None:
        decision_trace.append({
            "action": action,
            "reason": reason,
            "before": before,
            "after": after,
            "rule_source": "extract._apply_node_optimizations",
        })

    if name in merge_into_parent:
        parent_name = merge_into_parent[name]
        node["merged_into"] = parent_name
        node["decision_trace"] = decision_trace + [{
            "action": "merge_into_parent",
            "reason": f"normalized name matched merge map: {name}",
            "before": name,
            "after": parent_name,
            "rule_source": "extract._apply_node_optimizations",
        }]
        return None
    if name in layer_promotions:
        before = node.get("layer")
        node["layer"] = layer_promotions[name]
        record_decision("promote_layer", f"matched promotion map: {name}", before, node["layer"])
    if name in layer_downgrades:
        before = node.get("layer")
        node["layer"] = layer_downgrades[name]
        record_decision("demote_layer", f"matched downgrade map: {name}", before, node["layer"])
    alias_hit = _alias_lookup().get(name)
    if alias_hit:
        before_name = node.get("name") or node.get("meta_name")
        node["name"] = alias_hit["canonical"]
        node["meta_name"] = alias_hit["canonical"].lower()
        if alias_hit.get("meta_class") and node.get("meta_class") != alias_hit["meta_class"]:
            node["meta_class"] = alias_hit["meta_class"]
        record_decision("rewrite_name", f"matched alias map: {name}", before_name, node["name"])
    elif name in name_rewrites:
        before_name = node.get("name") or node.get("meta_name")
        node["name"] = name_rewrites[name]
        node["meta_name"] = name_rewrites[name].lower()
        record_decision("rewrite_name", f"matched rewrite map: {name}", before_name, node["name"])
    if name in attr_overrides:
        attrs = node.get("attributes") or []
        if isinstance(attrs, list):
            existing = {a.get("name") for a in attrs if isinstance(a, dict)}
            added_attrs = []
            for attr in attr_overrides[name]:
                if attr not in existing:
                    attrs.append({"name": attr, "value": ""})
                    added_attrs.append(attr)
            node["attributes"] = attrs
            if added_attrs:
                record_decision("add_attributes", f"matched attribute override map: {name}", None, added_attrs)
    if name == "supply chain" and meta_class == "Capability":
        before_class = node.get("meta_class")
        node["meta_class"] = "Capability"
        node["layer"] = "L1"
        record_decision("normalize_class", "capability supply chain canonicalization", before_class, node["meta_class"])
    if decision_trace:
        node["decision_trace"] = decision_trace
    return node

def _apply_node_optimizations(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(nodes, list):
        return nodes

    maps = _load_optimization_config()
    optimized = []
    for node in nodes:
        if not isinstance(node, dict):
            optimized.append(node)
            continue
        result = _optimize_node(node, **maps)
        if result is not None:
            optimized.append(result)
    return optimized

def _new_run_id(prefix: str = "UNIFIED") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

class _ExtractRunState:
    """Mutable holder for run_extract counters and accumulated payloads."""

    __slots__ = ("processed", "failed", "skipped", "nodes", "instances", "relationships", "fragment_paths", "file_quality")

    def __init__(self) -> None:
        self.processed = 0
        self.failed = 0
        self.skipped = 0
        self.nodes = []
        self.instances = []
        self.relationships = []
        self.fragment_paths = []
        self.file_quality = []

def _infer_document_type_from_source(source_path: Path) -> tuple[str, float, list[str], list[str]]:
    """Infer document type from sample text (best-effort, failures ignored)."""
    document_type = "unknown"
    document_type_confidence = 0.0
    secondary_types: list[str] = []
    document_rationale: list[str] = []
    try:
        sample_text = ""
        if source_path.is_file():
            sample_text = source_path.read_text(encoding="utf-8", errors="replace")[:12000]
        else:
            for f in list(source_path.rglob("*.md"))[:2] + list(source_path.rglob("*.txt"))[:2]:
                try:
                    sample_text += f.read_text(encoding="utf-8", errors="replace")[:4000]
                except Exception:
                    continue
        document_type, document_type_confidence, secondary_types, document_rationale = infer_document_type(sample_text)
    except Exception:
        pass
    return document_type, document_type_confidence, secondary_types, document_rationale

def _auto_detect_domain(source_path: Path) -> str:
    domain = "SUPPLY_CHAIN"
    document_analysis = None
    try:
        from ontology_compiler.services.extraction.prompt_service import DocumentAnalyzer
        analyzer = DocumentAnalyzer()
        if source_path.is_file():
            try:
                if source_path.suffix.lower() in [".pptx", ".ppt", ".docx", ".doc", ".pdf", ".xlsx", ".xls", ".csv"]:
                    text = FileParser.extract_text(str(source_path))
                else:
                    text = source_path.read_text(encoding="utf-8", errors="replace")
                document_analysis = analyzer.analyze(text)
                print(f"[Extract] Document analysis: type={document_analysis.get('document_type')}, topics={document_analysis.get('key_topics')}")
            except Exception as e:
                print(f"[Extract] Failed to analyze document: {e}")
        else:
            combined_text = ""
            for ext in [".pptx", ".ppt", ".docx", ".doc", ".pdf", ".md", ".txt", ".xlsx", ".xls", ".csv"]:
                for f in source_path.rglob(f"*{ext}"):
                    try:
                        if f.suffix.lower() in [".pptx", ".ppt", ".docx", ".doc", ".pdf", ".xlsx", ".xls", ".csv"]:
                            combined_text += FileParser.extract_text(str(f)) + "\n"
                        else:
                            combined_text += f.read_text(encoding="utf-8", errors="replace") + "\n"
                    except Exception as e:
                        print(f"[Extract] Failed to read file {f.name}: {e}")
                        continue
            if combined_text:
                document_analysis = analyzer.analyze(combined_text)
                print(f"[Extract] Document analysis: type={document_analysis.get('document_type')}, topics={document_analysis.get('key_topics')}")
    except ImportError as e:
        print(f"[Extract] DocumentAnalyzer not available: {e}, using keyword-based domain detection only")

    inferred_domain, scores = _infer_domain_from_files(source_path, document_analysis)
    if inferred_domain:
        domain = inferred_domain
        print(f"[Extract] Auto-detected domain: {domain} (scores: {scores})")
    else:
        print(f"[Extract] Domain auto-detection failed, using default: {domain}")
    return domain

def _collect_input_files(source_path: Path) -> list[Path]:
    """Collect supported source files (single file, or recursive dir scan)."""
    if source_path.is_dir():
        input_files = []
        for ext in [".pptx", ".ppt", ".docx", ".doc", ".pdf", ".md", ".txt", ".xlsx", ".xls", ".csv", ".json"]:
            input_files.extend(source_path.rglob(f"*{ext}"))
        return input_files
    return [source_path]

def _load_skill_hits(source_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load L25 skill matches for the source; returns (skill_hits, skill_stats)."""
    if not _L25_SKILLS_AVAILABLE:
        return [], {"error": "l25 skills unavailable"}
    try:
        skill_loader = get_l25_skill_loader()
        skill_stats = skill_loader.get_stats()
        preview_text = ""
        if source_path.is_file():
            try:
                suffix = source_path.suffix.lower()
                if suffix in {".pptx", ".ppt", ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv"}:
                    preview_text = FileParser.extract_text(str(source_path))[:12000]
                else:
                    preview_text = source_path.read_text(encoding="utf-8", errors="replace")[:12000]
            except Exception:
                preview_text = ""
        matched_skills = skill_loader.find_matching_skills(preview_text) if preview_text else []
        skill_hits = [
            {
                "skill_id": getattr(skill, "skill_id", ""),
                "skill_name": getattr(skill, "skill_name", getattr(skill, "name", "")),
                "domain": getattr(skill, "domain", []),
                "priority": getattr(skill, "priority", 0),
            }
            for skill in matched_skills
        ]
        return skill_hits, skill_stats
    except Exception as e:
        return [], {"error": str(e)}

def _build_document_metadata(
    file_path: Path,
    document_type: str,
    document_type_confidence: float,
    secondary_types: list[str],
    document_rationale: list[str],
    skill_hits: list[dict[str, Any]],
    skill_stats: dict[str, Any],
) -> dict[str, Any]:
    """Build the per-file document_metadata payload passed to the extractor."""
    try:
        page_hint = None
        if file_path.suffix.lower() == ".pdf":
            page_hint = (1, 5)
        chunk_strategy = _infer_chunk_strategy(file_path.suffix.lower(), document_type)
        return {
            "source_path": str(file_path),
            "file_suffix": file_path.suffix.lower(),
            "page_hint": page_hint,
            "document_type": document_type,
            "document_type_confidence": document_type_confidence,
            "secondary_types": secondary_types,
            "document_rationale": document_rationale,
            "matched_skills": skill_hits,
            "skill_stats": skill_stats,
            "chunk_strategy": chunk_strategy,
        }
    except Exception:
        return {}

def _infer_chunk_strategy(file_suffix: str, document_type: str) -> str:
    """Choose an explicit chunking strategy for a source file.

    PPT / slide documents prefer slide boundaries; markdown/docs prefer
    semantic section chunking; tabular data prefers fixed-size chunks.
    """
    suffix = (file_suffix or "").lower()
    if suffix in {".pptx", ".ppt"}:
        return "slide_based"
    if suffix in {".md", ".markdown", ".docx", ".doc"}:
        return "section_based"
    if suffix in {".xlsx", ".xls", ".csv"}:
        return "fixed"
    return "hybrid"

def _consume_cache_hit(
    file_path: Path,
    file_root: Path,
    file_run_id: str,
    cache,
    cached,
    state: "_ExtractRunState",
    version_bundle: Any = None,
) -> bool:
    """Replay a cached fragment into state; True when the file was served from cache."""
    served_from = "versioned" if version_bundle is not None and not version_bundle.is_empty() else "unversioned"
    print(f"[Extract] [SKIP] File unchanged: {file_path.name} (cached at {cached['processed_at']}, {served_from} cache)", flush=True)

    cached_fragment_path = file_root / f"fragment_{file_run_id}.json"
    fragment_data = None
    fragment_source_path = None
    if cached_fragment_path.exists():
        fragment_data = _read_json(cached_fragment_path, {})
        fragment_source_path = cached_fragment_path
    else:
        cache_text_path = cache.text_store / f"{cached.get('text_key')}.json"
        if cache_text_path.exists():
            fragment_data = _read_json(cache_text_path, {})
            fragment_source_path = cache_text_path

    if fragment_data is not None:
        if not fragment_has_content(fragment_data):
            print(f"[Extract] [CACHE MISS] Empty cached fragment for {file_path.name}; reprocessing", flush=True)
            # The cached entry is polluted (e.g. written before the empty-result
            # guard existed). Invalidate it so subsequent runs do not repeatedly
            # hit-then-reprocess, and clear the stale payload on disk.
            try:
                cache.invalidate_file(file_path)
            except Exception:
                pass
            return False
        fragment_data = _annotate_fragment_entities(fragment_data, str(file_path), file_run_id)
        # Persist a fresh run-local fragment so downstream stages consume this
        # run's payload instead of a stale global cache file (which may predate
        # newer safety-net / relation rules). The global cache text store is a
        # historical snapshot, not a live run artifact.
        try:
            _write_json(cached_fragment_path, fragment_data)
            fragment_source_path = cached_fragment_path
        except Exception:
            pass
        # 保存Markdown格式的原始文件（缓存命中时也生成一份）
        try:
            markdown_content = FileParser.extract_markdown(str(file_path))
            markdown_file_path = file_root / f"source_{file_run_id}.md"
            markdown_file_path.write_text(markdown_content, encoding="utf-8", errors="replace")
            logging.info(f"[Extract] [CACHE] 已保存Markdown格式原始文件: {markdown_file_path}")
        except Exception as e:
            logging.warning(f"[Extract] [CACHE] Markdown渲染失败: {e}")
        state.nodes.extend(fragment_data.get("nodes", []))
        state.instances.extend(fragment_data.get("instances", []))
        state.relationships.extend(fragment_data.get("relationships", []))
        state.fragment_paths.append(str(fragment_source_path))
        state.processed += 1
        return True

    print(f"[Extract] [CACHE MISS] Metadata hit but fragment payload missing: {file_path.name}; reprocessing", flush=True)
    return False

def _persist_success_fragment(
    file_path: Path,
    idx: int,
    file_root: Path,
    file_run_id: str,
    cache,
    state: "_ExtractRunState",
    result,
    version_bundle: Any = None,
) -> None:
    """Annotate, accumulate into state, and persist a successful extraction fragment."""
    normalized_file_root = file_root
    fragment_path = normalized_file_root / f"fragment_{file_run_id}.json"
    print(f"[Extract][DEBUG] normalized_file_root[{idx}]={normalized_file_root}", flush=True)
    print(f"[Extract][DEBUG] fragment_path[{idx}]={fragment_path}", flush=True)

    # 2026-10-13: TrueUnifiedExtractor 返回扁平 dict（nodes/relationships 顶层），
    # 无 'data' 包装。兼容两种形状，防 28 nodes 被当空丢弃。
    fragment_data = result.get('data', {}) or {}
    if not fragment_has_content(fragment_data) and fragment_has_content(result):
        fragment_data = {k: result.get(k) for k in ("nodes", "relationships", "instances") if result.get(k)}
        for k, v in (result.get('metadata') or {}).items():
            fragment_data.setdefault(k, v)
    fragment_data = _annotate_fragment_entities(fragment_data, str(file_path), file_run_id)
    state.nodes.extend(fragment_data.get("nodes", []))
    state.instances.extend(fragment_data.get("instances", []))
    state.relationships.extend(fragment_data.get("relationships", []))

    quality_entry = {
        "file": file_path.name,
        "extraction_quality_status": fragment_data.get("extraction_quality_status", "unknown"),
        "validation_passed": fragment_data.get("validation_passed", False),
        "quality_score": fragment_data.get("quality_score", 0.0),
        "chunk_quality": fragment_data.get("chunk_quality", ""),
        "chunk_metrics": fragment_data.get("chunk_metrics", []),
    }
    state.file_quality.append(quality_entry)

    _write_json(fragment_path, fragment_data)

    if cache and fragment_has_content(fragment_data):
        cleaned_text = json.dumps(fragment_data, ensure_ascii=False)[:10000]
        cache.cache_result(file_path, cleaned_text,
            len(fragment_data.get("nodes", [])), len(fragment_data.get("relationships", [])),
            result_data=fragment_data,
            version_bundle=version_bundle if version_bundle is not None else None)
        print(f"[Extract] [CACHED] File processed and cached: {file_path.name}", flush=True)
    elif cache:
        print(f"[Extract] [NOT CACHED] Empty/degraded result for {file_path.name}; skipping cache", flush=True)
    state.fragment_paths.append(str(fragment_path))
    state.processed += 1
    print(f"[Extract] Completed: {file_path.name} ({len(fragment_data.get('nodes', []))} nodes, {len(fragment_data.get('relationships', []))} rels)", flush=True)

async def _process_extract_file(
    file_path: Path,
    idx: int,
    *,
    state: "_ExtractRunState",
    run_id: str,
    file_root_by_index,
    cache,
    semaphore,
    document_type: str,
    document_type_confidence: float,
    secondary_types: list[str],
    document_rationale: list[str],
    skill_hits: list[dict[str, Any]],
    skill_stats: dict[str, Any],
    domain: str,
    version_bundle=None,
) -> None:
    """Process one extract source file: cache-hit fast path or full compile."""
    from ontology_compiler import compile_source_file
    async with semaphore:
        file_run_id = f"{run_id}-{idx:04d}"
        file_root = file_root_by_index(idx)
        file_output_dir = file_root
        print(f"[Extract][DEBUG] file_output_dir[{idx}]={file_output_dir}")
        file_output_dir.mkdir(parents=True, exist_ok=True)

        if cache:
            cached = cache.get_cached_result(
                file_path,
                version_bundle=version_bundle if version_bundle is not None else None,
            )
            if cached and _consume_cache_hit(file_path, file_root, file_run_id, cache, cached, state, version_bundle):
                return

        try:
            print(f"[Extract] Starting processing: {file_path.name}", flush=True)
            # 渲染原始文件为Markdown格式并保存
            try:
                markdown_content = FileParser.extract_markdown(str(file_path))
                # 保存Markdown文件到本地目录
                markdown_file_path = file_root / f"source_{file_run_id}.md"
                markdown_file_path.write_text(markdown_content, encoding="utf-8", errors="replace")
                logging.info(f"[Extract] 已保存Markdown格式原始文件: {markdown_file_path}")
            except Exception as e:
                logging.warning(f"[Extract] Markdown渲染失败，将使用原始文本: {e}")
                markdown_content = None
            
            document_metadata = _build_document_metadata(
                file_path,
                document_type,
                document_type_confidence,
                secondary_types,
                document_rationale,
                skill_hits,
                skill_stats,
            )
            # 如果Markdown渲染成功，使用Markdown内容作为输入
            if markdown_content:
                # 替换为Markdown内容
                from ontology_compiler.services.extraction.unified_compilation_pipeline import CompilationRequest
                request = CompilationRequest(
                    source_path=str(file_path),
                    domain=domain,
                    document_metadata=document_metadata,
                    fast_mode=False,
                    run_id=run_id,
                )
                # 直接使用Markdown内容调用extractor
                from ontology_compiler.services.extraction.true_unified_extractor import TrueUnifiedExtractor
                chunk_strategy = (document_metadata.get("chunk_strategy") or "").strip() or None
                extractor = TrueUnifiedExtractor(chunk_strategy=chunk_strategy)
                result = await asyncio.wait_for(
                    extractor.extract_async_content(
                        content=markdown_content,
                        domain=domain,
                        source_file=str(file_path),
                        fast_mode=False,
                        document_metadata=document_metadata,
                    ),
                    timeout=_file_timeout_seconds(),
                )
            else:
                # 回退到原始文本
                result = await asyncio.wait_for(
                    compile_source_file(str(file_path), domain=domain, document_metadata=document_metadata),
                    timeout=_file_timeout_seconds(),
                )
            # 2026-10-13: TrueUnifiedExtractor 返回 dict 无顶层 status，
            # 成功结果被误判 FAILED（40 nodes 产物丢弃）。先读顶层，再读
            # metadata.extraction_quality_status（passed/partial/failed）。
            result_status = str(result.get('status') or '').upper()
            if not result_status:
                qs = str(((result.get('metadata') or {}).get('extraction_quality_status')) or '').lower()
                result_status = {'passed': 'SUCCESS', 'partial': 'DEGRADED_SUCCESS'}.get(qs, '')
            if result_status == 'SUCCESS':
                _persist_success_fragment(file_path, idx, file_root, file_run_id, cache, state, result, version_bundle)
            elif result_status == 'DEGRADED_SUCCESS':
                state.processed += 1
                print(f"[Extract] [DEGRADED] {file_path.name}: {result.get('error') or result.get('message') or 'degraded result'}", flush=True)
            else:
                state.failed += 1
                print(f"[Extract] [FAILED] {file_path.name}: {result.get('error') or result.get('message') or result_status}", flush=True)
        except asyncio.TimeoutError:
            print(f"[Extract] [TIMEOUT] File processing timed out: {file_path.name}", flush=True)
            state.failed += 1
        except Exception as e:
            print(f"[Extract] [ERROR] {file_path.name}: {e}", flush=True)
            state.failed += 1

async def run_extract(args: Namespace) -> StageResult:
    start = time.time()
    from ontology_compiler import compile_source_file

    # C-4: resolve the versioned cache bundle once per run so the document
    # cache key carries prompt/schema/policy/model/mode dimensions. A version
    # bump invalidates every cached fragment instead of silently reusing a
    # stale result.
    version_bundle = None
    try:
        from src.pipeline.versioned_cache import resolve_version_bundle
        version_bundle = resolve_version_bundle(
            extraction_mode=os.environ.get("PIPELINE_EXECUTION_MODE", "production"),
        )
    except Exception:
        version_bundle = None

    source = args.file or args.folder
    if not source:
        return StageResult("extract", False, "extract requires --file or --folder", elapsed_seconds=round(time.time() - start, 3))

    domain = getattr(args, "domain", "AUTO")
    run_id = getattr(args, "run_id", None) or _new_run_id()
    planned_paths = build_pipeline_run_paths(
        getattr(args, "run_root", args.output_dir),
        run_id,
        stage_name="extract",
        file_index=0,
    )
    output_dir = planned_paths.run_root
    print(f"[Extract][DEBUG] root_output_dir={output_dir} raw_output_dir={args.output_dir} run_id={run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _log_unified_config_diagnostics()

    file_root_by_index = planned_paths.file_root_for_index

    source_path = Path(source)
    if not source_path.exists():
        return StageResult("extract", False, f"source not found: {source}", elapsed_seconds=round(time.time() - start, 3))

    document_type, document_type_confidence, secondary_types, document_rationale = _infer_document_type_from_source(source_path)

    if domain == "AUTO":
        domain = _auto_detect_domain(source_path)

    input_files = _collect_input_files(source_path)
    if not input_files:
        return StageResult("extract", False, f"no supported files found in {source}", elapsed_seconds=round(time.time() - start, 3))

    total = len(input_files)
    skill_hits, skill_stats = _load_skill_hits(source_path)

    state = _ExtractRunState()
    cache = DocumentCache() if _CACHE_AVAILABLE else None
    if cache:
        print("[Extract] Document cache enabled")
    else:
        print("[Extract] Document cache not available")

    semaphore = asyncio.Semaphore(_file_concurrency())
    await asyncio.gather(*[
        _process_extract_file(
            f,
            i,
            state=state,
            run_id=run_id,
            file_root_by_index=file_root_by_index,
            cache=cache,
            semaphore=semaphore,
            document_type=document_type,
            document_type_confidence=document_type_confidence,
            secondary_types=secondary_types,
            document_rationale=document_rationale,
            skill_hits=skill_hits,
            skill_stats=skill_stats,
            domain=domain,
            version_bundle=version_bundle,
        )
        for i, f in enumerate(input_files)
    ])

    _extract_end = time.time()
    print(f"[Extract] \u23f1\ufe0f File processing elapsed: {round(_extract_end - start, 1)}s")

    _enable_candidate_gen = os.environ.get("EXTRACT_ENABLE_CANDIDATE_GEN", "0") == "1"
    if _LLM_CANDIDATE_GENERATOR_AVAILABLE and state.nodes and _enable_candidate_gen:
        print(f"[Extract] [LLM Candidate Generator] Processing {len(state.nodes)} nodes for candidate enhancement")
        try:
            try:
                from src.application.bootstrap.llm_bootstrap import get_llm_client
                llm_client = get_llm_client()
            except Exception:
                llm_client = None
            budget_policy = None
            try:
                from src.application.services.prompt_budget_policy import get_prompt_budget_policy
                budget_policy = get_prompt_budget_policy()
            except Exception:
                pass
            candidate_generator = LLMCandidateGenerator(
                llm_client=llm_client,
                budget_policy=budget_policy,
            )
            combined_text = ""
            for node in state.nodes[:10]:
                combined_text += json.dumps(node, ensure_ascii=False) + "\n"

            scene_result = await candidate_generator.generate_scene_classification(combined_text, domain_hint=domain, document_type=document_type)
            print(f"[Extract] [LLM Candidate Generator] Scene classification: {scene_result.status.value}")

            entity_types = list({n.get("meta_class", "Concept") for n in state.nodes[:10] if isinstance(n, dict)})
            entity_result = await candidate_generator.generate_entity_candidates(combined_text, entity_types=entity_types, domain_hint=domain, document_type=document_type)
            print(f"[Extract] [LLM Candidate Generator] Entity candidates: {len(entity_result.candidates)} generated")

            relation_result = await candidate_generator.generate_relation_candidates(combined_text, entities=state.nodes[:10], domain_hint=domain, document_type=document_type)
            print(f"[Extract] [LLM Candidate Generator] Relation candidates: {len(relation_result.candidates)} generated")

            append_candidate_entities(state, entity_result.candidates, str(source_path), run_id)
            append_candidate_relationships(state, relation_result.candidates, str(source_path), run_id)
            for candidate in entity_result.candidates[:5]:
                print(f"[Extract] [LLM Candidate Generator] Candidate: {candidate.type} - {candidate.name} (confidence: {candidate.confidence})")
        except Exception as e:
            print(f"[Extract] [LLM Candidate Generator] Enhancement skipped: {e}")

    all_nodes = _normalize_entity_collection(state.nodes, str(source_path), run_id, "extract")
    all_nodes = _apply_node_optimizations(all_nodes)
    # _normalize_l0_entities returns (l0, rest); flatten both into one node list.
    # (Was assigned directly before, producing nested [[l0], [rest]] that
    # downstream consumers could not read.)
    _l0_nodes, _non_l0_nodes = _normalize_l0_entities(all_nodes)
    all_nodes = _l0_nodes + _non_l0_nodes
    all_nodes = _apply_global_id_uniqueness(all_nodes, state.relationships)
    _apply_l25_rule_safety_net(all_nodes)
    _apply_meta_class_safety_net(all_nodes)
    all_nodes = ensure_deterministic_domain_entities(all_nodes)
    # Backfilled canonical entities (BOM, Inventory, ...) carry no meta_id;
    # assign one so merge can resolve relation endpoints to real IDs.
    all_nodes = _assign_missing_entity_ids(all_nodes)
    # before optimizations; re-normalizing would undo alias rewrites.

    all_instances = _normalize_entity_collection(state.instances, str(source_path), run_id, "extract")
    all_relationships = _normalize_relationship_collection(state.relationships, str(source_path), run_id, "extract")
    all_relationships = append_deterministic_domain_relations(all_nodes, all_relationships, str(source_path), run_id)
    all_relationships = _apply_relationship_governance(all_relationships, domain)

    if not all_nodes and not all_instances and not all_relationships:
        print("[Extract] No entities or relationships extracted after normalization")

    metadata_payload = {
        "document_type": document_type,
        "document_type_confidence": document_type_confidence,
        "secondary_types": secondary_types,
        "document_rationale": document_rationale,
        "skill_hits": skill_hits,
        "skill_stats": skill_stats,
    }

    extraction_quality_status = "passed"
    if all_nodes or all_instances or all_relationships:
        pass
    elif state.failed > 0:
        extraction_quality_status = "failed"
    file_level_quality = state.file_quality if state.file_quality else [{
        "file": str(source_path),
        "extraction_quality_status": "passed" if (all_nodes or all_instances or all_relationships) else "failed",
        "validation_passed": bool(all_nodes or all_instances or all_relationships),
        "quality_score": 0.0,
        "chunk_quality": "",
        "chunk_metrics": [],
    }]
    extraction_quality = {
        "status": extraction_quality_status,
        "validation_passed": any(item.get("validation_passed") for item in file_level_quality),
        "files": file_level_quality,
        "entity_count": len(all_nodes) + len(all_instances),
        "relation_count": len(all_relationships),
    }

    success = state.processed > 0 and state.failed == 0
    if state.failed > 0 and state.processed == 0:
        success = False
    if state.skipped > 0 and state.processed == 0 and state.failed == 0:
        success = False

    # Persist the complete post-safety-net aggregate as the run's combined
    # fragment at the run root (the layout `_resolve_merge_inputs` expects for
    # "fragment_*.json"). Prepending it to fragment_paths makes
    # resolve_stage_fragment feed merge this fresh aggregate instead of a stale
    # per-file cache snapshot (whose relations predate the safety net).
    try:
        combined_fragment_path = output_dir / f"fragment_{run_id}.json"
        _write_json(combined_fragment_path, {
            "nodes": all_nodes,
            "instances": all_instances,
            "relationships": all_relationships,
            "metadata": {
                "run_id": run_id,
                "stage": "extract",
                "combined": True,
                "source_path": str(source_path),
                "domain": domain,
                "entity_count": len(all_nodes) + len(all_instances),
                "relation_count": len(all_relationships),
            },
        })
        if str(combined_fragment_path) not in state.fragment_paths:
            state.fragment_paths.insert(0, str(combined_fragment_path))
    except Exception as exc:
        print(f"[Extract] [WARN] Failed to persist combined fragment: {exc}", flush=True)

    result = StageResult(
        "extract",
        success,
        f"processed {state.processed}/{total} files (skipped {state.skipped}, failed {state.failed})",
        {
            "run_id": run_id,
            "source_path": str(source_path),
            "domain": domain,
            "document_metadata": metadata_payload,
            "extraction_quality": extraction_quality,
            "nodes": all_nodes,
            "instances": all_instances,
            "relationships": all_relationships,
            "fragment_paths": state.fragment_paths,
        },
        elapsed_seconds=round(time.time() - start, 3),
        artifact_paths=state.fragment_paths,
    )

    # M4 (CQ_PIPELINE_IMPLEMENTATION_PLAN): CQ 关注词表 coverage + 漏抽提示。
    # 纯只读统计，不改抽取结果、不参与候选打分（CQ 只做约束/提示）；无 CQ
    # 输入时 helper 返回 {}，stage 行为与无 CQ 时一致。
    try:
        from src.pipeline.cq_integration import extract_cq_stage_hint

        result.data.update(extract_cq_stage_hint(all_nodes, all_instances, all_relationships, getattr(args, "artifact_root", None)))
    except Exception:
        pass

    return result
