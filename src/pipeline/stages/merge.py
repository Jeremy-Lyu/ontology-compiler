"""Merge and Dedup stages."""
import os
import shutil
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .common import StageResult, PROJECT_ROOT, PIPELINE_VALIDATION_THRESHOLDS, build_backup_metadata, _read_json, _write_json, _latest_file, _normalize_fragment, _entity_layer_value
from ontology_compiler.services.extraction.entity_normalizer import normalize_status

import re

def _entity_signature(node: dict[str, Any], include_source: bool = True) -> tuple[str, ...]:
    if not isinstance(node, dict):
        return ("",)
    layer = str(node.get("layer") or node.get("layer_type") or "").strip()
    name = str(node.get("name") or node.get("meta_name") or node.get("scenario_name") or "").strip().lower()
    meta_class = str(node.get("meta_class") or node.get("entity_type") or "").strip().lower()
    variant = str(node.get("variant") or node.get("variant_code") or node.get("variant_designation") or "").strip().lower()
    position = str(node.get("position") or node.get("position_code") or node.get("position_label") or "").strip().lower()
    occurrence = str(node.get("occurrence") or node.get("occurrence_id") or "").strip().lower()
    source_parts = []
    if include_source:
        for key in ("source_structure", "source_file", "source_document", "provenance"):
            val = node.get(key)
            if isinstance(val, dict):
                source_parts.append(str(val.get("source_file") or val.get("source_id") or val.get("source_type") or "").strip().lower())
            elif val:
                source_parts.append(str(val).strip().lower())
    return (layer, name, meta_class, variant, position, occurrence, *source_parts)

def _keep_entity_provenance(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(target, dict) or not isinstance(source, dict):
        return target
    for key in ("provenance", "source_file", "source_structure", "source_document", "variant", "variant_code", "variant_designation", "position", "position_code", "position_label", "occurrence", "occurrence_id"):
        if key in source and key not in target:
            target[key] = source[key]
    if isinstance(source.get("evidence"), dict) and not isinstance(target.get("evidence"), dict):
        target["evidence"] = source["evidence"]
    return target

PREFIXES_TO_CAPITALIZE = {
    'API', 'ERP', 'MBOM', 'SBOM', 'PLM', 'BOM', 'KDP', 'VSM',
    'RFID', 'CD', 'VCC', 'CRM', 'SCM', 'WMS', 'MES', 'CAD', 'CAE'
}

def _normalize_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return name
    
    name = name.strip()
    if not name:
        return name
    
    name = re.sub(r'[^\w\s\-]', '', name)
    
    words = name.split()
    normalized_words = []
    for word in words:
        if word.upper() in PREFIXES_TO_CAPITALIZE:
            normalized_words.append(word.upper())
        elif word.isupper() and len(word) > 1:
            normalized_words.append(word)
        else:
            normalized_words.append(word.capitalize())
    
    return ' '.join(normalized_words)

STANDARD_META_CLASSES = {"Capability", "System", "Actor", "Asset", "Data", "Process"}

META_CLASS_MAPPING = {
    "Concept": "Data",
    "BusinessScenario": "System",
    "MappingRule": "Data",
    "BusinessRule": "Data",
    "ValidationRule": "Data",
    "BusinessObject": "Asset",
    "StorageLocation": "Asset",
    "Instance": "Asset",
    "Domain": "Capability",
    "SubDomain": "Capability",
    "Actionable": "Capability",
    "Equipment_Infra": "Actor",
    "Document": "Data",
    "Structure": "System",
    "DataEntity": "Data",
}

def _normalize_meta_class(meta_class: str) -> str:
    """Normalize meta_class to one of the 6 standard main classes."""
    if not meta_class:
        return "System"
    mc = meta_class.strip()
    if mc in STANDARD_META_CLASSES:
        return mc
    return META_CLASS_MAPPING.get(mc, "System")

def _normalize_entity_for_merge(entity: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entity, dict):
        return entity
    
    entity['status'] = normalize_status(entity.get('status'))
    
    entity_id = entity.get('entity_id', entity.get('meta_id', ''))
    if 'layer' not in entity or not entity['layer']:
        if entity_id.startswith('L0'):
            entity['layer'] = 'L0'
        elif entity_id.startswith('L1'):
            entity['layer'] = 'L1'
        elif entity_id.startswith('L25'):
            entity['layer'] = 'L2.5'
        elif entity_id.startswith('L3'):
            entity['layer'] = 'L3'
        elif entity_id.startswith('L2'):
            entity['layer'] = 'L2'
        else:
            entity['layer'] = 'L1'
    
    if 'name' in entity and isinstance(entity['name'], str):
        entity['name'] = _normalize_name(entity['name'])
    elif 'meta_name' in entity and isinstance(entity['meta_name'], str):
        entity['name'] = _normalize_name(entity['meta_name'])
    
    if 'meta_name' in entity and isinstance(entity['meta_name'], str):
        entity['meta_name'] = _normalize_name(entity['meta_name'])
    elif 'name' in entity and isinstance(entity['name'], str):
        entity['meta_name'] = _normalize_name(entity['name'])
    
    if 'scenario_name' in entity and isinstance(entity['scenario_name'], str):
        entity['scenario_name'] = _normalize_name(entity['scenario_name'])
    
    entity_type = entity.get('entity_type', entity.get('meta_class', ''))
    if 'meta_class' not in entity or not entity['meta_class']:
        entity['meta_class'] = _normalize_meta_class(entity_type if entity_type else "System")
    else:
        entity['meta_class'] = _normalize_meta_class(entity['meta_class'])
    
    if 'entity_type' not in entity or not entity['entity_type']:
        entity['entity_type'] = entity['meta_class']
    
    if 'entity_id' not in entity or not entity['entity_id']:
        entity['entity_id'] = entity_id

    if 'origin_kind' not in entity:
        entity['origin_kind'] = 'derived'
    if 'origin_run_id' not in entity and entity.get('run_id'):
        entity['origin_run_id'] = entity['run_id']
    
    if 'description' not in entity or not entity['description']:
        name = entity.get('meta_name', '') or entity.get('name', '')
        if name:
            entity['description'] = f"A {entity.get('meta_class', 'Concept')} representing {name}."
        else:
            entity['description'] = "Description not available."
    else:
        desc = str(entity['description']).strip()
        if len(desc) < 10:
            name = entity.get('meta_name', '') or entity.get('name', '')
            if name:
                entity['description'] = f"A {entity.get('meta_class', 'Concept')} representing {name}."
            else:
                entity['description'] = "Description not available."
    
    if 'domain' not in entity or not entity['domain']:
        entity['domain'] = "SUPPLY_CHAIN"
    
    if entity.get('layer') == 'L3':
        entity['entity_type'] = "Instance"
        entity['meta_class'] = _normalize_meta_class(entity.get('meta_class', 'Asset'))
    
    return entity

def _normalize_all_entities(nodes: List[Dict[str, Any]], instances: List[Dict[str, Any]]) -> tuple:
    """Normalize all entities in merge result."""
    normalized_nodes = [_normalize_entity_for_merge(n) for n in nodes]
    normalized_instances = [_normalize_entity_for_merge(i) for i in instances]
    return normalized_nodes, normalized_instances

def _entity_signature(entity: Dict[str, Any], include_source: bool = True) -> tuple:
    """Build a domain-aware signature for deduplication.

    MBOM and manufacturing documents often reuse the same entity name across
    different variants, positions, or source structures. Include provenance-
    relevant fields so we do not collapse semantically distinct objects.
    """
    if not isinstance(entity, dict):
        return ("invalid",)

    layer = str(entity.get("layer", "L1")).strip()
    name = str(entity.get("name") or entity.get("meta_name") or "").strip().lower()
    meta_class = str(entity.get("meta_class") or entity.get("entity_type") or "").strip().lower()
    variant = str(
        entity.get("variant")
        or entity.get("variant_code")
        or entity.get("variant_name")
        or entity.get("variant_designation")
        or entity.get("properties", {}).get("variant", "")
    ).strip().lower()
    position = str(
        entity.get("position")
        or entity.get("position_code")
        or entity.get("position_label")
        or entity.get("occurrence", {}).get("position", "")
    ).strip().lower()
    occurrence = str(
        entity.get("occurrence_id")
        or entity.get("occurrence")
        or entity.get("meta_occurrence_id")
        or entity.get("properties", {}).get("occurrence_id", "")
    ).strip().lower()
    evidence = entity.get("evidence")
    evidence_source = evidence.get("source_file", "") if isinstance(evidence, dict) else ""
    source_structure = str(
        entity.get("source_structure")
        or entity.get("source_id")
        or entity.get("source_document")
        or entity.get("source_file")
        or evidence_source
    ).strip().lower()

    signature = [layer, name, meta_class]
    if variant:
        signature.append(f"variant:{variant}")
    if position:
        signature.append(f"position:{position}")
    if occurrence:
        signature.append(f"occurrence:{occurrence}")
    if include_source and source_structure:
        signature.append(f"source:{source_structure}")
    return tuple(signature)

def _keep_entity_provenance(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge lightweight provenance markers when deduplicating entities."""
    if not isinstance(base, dict) or not isinstance(incoming, dict):
        return base

    provenance = base.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
        base["provenance"] = provenance

    incoming_prov = incoming.get("provenance", {})
    if isinstance(incoming_prov, dict):
        for key in ("source_file", "source_structure", "run_id", "rule_version", "model_version", "skill_version"):
            value = incoming_prov.get(key) or incoming.get(key)
            if value and key not in provenance:
                provenance[key] = value

    if incoming.get("evidence") and not base.get("evidence"):
        base["evidence"] = incoming["evidence"]

    for field in ("source_file", "source_structure", "source_document"):
        if incoming.get(field) and not base.get(field):
            base[field] = incoming[field]

    for field in ("variant", "variant_code", "variant_name", "variant_designation", "position", "position_code", "position_label", "occurrence", "occurrence_id"):
        if incoming.get(field) and not base.get(field):
            base[field] = incoming[field]

    return base

try:
    from src.pipeline.storage.master_store import MasterOntologyStore, create_master_store
except ImportError:
    MasterOntologyStore = None
    create_master_store = None

def migrate_legacy_master(master: dict[str, Any]) -> dict[str, Any]:
    try:
        from scripts.legacy_migration import migrate_legacy_master as _migrate
        return _migrate(master)
    except ImportError:
        try:
            from legacy_migration import migrate_legacy_master as _migrate
            return _migrate(master)
        except ImportError:
            return master

# ---------------------------------------------------------------------------
# Merge helpers — extracted from run_merge (Phase 3) so the orchestrator stays
# thin. Behavior is locked by tests/pipeline/test_merge_characterization.py.
# ---------------------------------------------------------------------------

# [L1->L3 MIGRATION]: patterns and known system instances moved to module scope.
INSTANCE_PATTERNS = [
    re.compile(r'^CPT\d+', re.IGNORECASE),  # CPT311, CPT034, etc.
    re.compile(r'^\w+-\d+$'),  # System-123 pattern
]
KNOWN_INSTANCES = {
    'CPT311', 'CPT034', 'CPT140', 'CPT950', 'CPT309', 'CPT310',
    'BIS', 'XPS', 'EcoEMOS', 'ANDON', 'ATACQ', 'MES',
    'AGVSystem', 'HofmannMasterPLC',
}
MBOM_KEEP_NAMES = {
    "delivery unit", "deliveryunit", "product mbom", "plant mbom", "plant bop",
    "ebom", "design bom", "position in bom", "position", "occurrence", "variant",
}

def _resolve_merge_inputs(args: Namespace) -> tuple[Path | None, Path, Path | None]:
    """Resolve fragment + master + run_dir paths for a merge run."""
    output_dir = Path(args.output_dir)
    run_dir = Path(args.run_dir) if getattr(args, "run_dir", None) else None
    if run_dir and not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    fragment_artifact = getattr(args, "fragment", None)
    if fragment_artifact:
        fragment_path = Path(fragment_artifact)
    elif run_dir:
        fragment_path = _latest_file(run_dir, "fragment_*.json")
    else:
        # Extract writes the combined fragment to the run root as
        # ``fragment_*.json``; older layouts placed it one level deep under a
        # subdirectory. Look in both locations so merge is robust to either.
        fragment_path = (
            _latest_file(output_dir, "*/fragment_*.json")
            or _latest_file(output_dir, "fragment_*.json")
        )

    master_path = Path(args.master_path)
    return fragment_path, master_path, run_dir

def _load_master(master_path: Path) -> dict[str, Any]:
    """Load the master ontology (store-backed or plain JSON) and migrate legacy shapes."""
    if MasterOntologyStore and master_path.exists():
        store = create_master_store(str(master_path), sync_mode=True)
        master = store.read()
    else:
        master = _read_json(master_path, {})

    has_legacy = any(master.get(k) for k in ["layer1_concepts", "layer2_scenarios", "layer2_logic", "layer2_definitions", "layer25_rules", "layer25_mapping_rules"])
    if has_legacy and not master.get("metadata", {}).get("legacy_migrated"):
        master = migrate_legacy_master(master)
    return master

def _load_fragment(fragment_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the fragment JSON and its normalized node/instance/relationship view."""
    if fragment_path.parent.name == "files":
        fragment_path = fragment_path.parent.parent / fragment_path.name
    fragment = _read_json(fragment_path, {})
    normalized = _normalize_fragment(fragment)
    return fragment, normalized

def _collect_legacy_nodes(master: dict[str, Any], existing_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold legacy layer entities that are not yet in nodes[] into existing_nodes."""
    legacy_layer_keys = ["layer1_concepts", "layer2_scenarios", "layer25_mapping_rules", "layer0_metadata"]
    for key in legacy_layer_keys:
        if key in master and isinstance(master[key], list):
            for entity in master[key]:
                if isinstance(entity, dict):
                    eid = str(entity.get("meta_id") or entity.get("entity_id") or entity.get("id") or "")
                    if eid:
                        existing_ids = {str(n.get("meta_id") or n.get("entity_id") or n.get("id") or "") for n in existing_nodes}
                        if eid not in existing_ids:
                            existing_nodes.append(entity)
    return existing_nodes

def _collect_existing_ids(
    existing_nodes: list[dict[str, Any]],
    existing_instances: list[dict[str, Any]],
) -> set[str]:
    """Collect all IDs already present in the master (nodes + instances)."""
    existing_id_set: set[str] = set()
    for _n in existing_nodes + existing_instances:
        if isinstance(_n, dict):
            _eid = str(_n.get("meta_id") or _n.get("entity_id") or _n.get("id") or "")
            if _eid:
                existing_id_set.add(_eid)
    return existing_id_set

def _compute_layer_max(
    existing_nodes: list[dict[str, Any]],
    existing_instances: list[dict[str, Any]],
    normalized: dict[str, Any],
) -> dict[str, int]:
    """Build layer prefix → max numeric suffix mapping for generating new unique IDs."""
    from collections import defaultdict as _dd
    _layer_max: dict[str, int] = _dd(int)
    for _n in existing_nodes + existing_instances + normalized["nodes"] + normalized.get("instances", []):
        if not isinstance(_n, dict):
            continue
        _eid = str(_n.get("meta_id") or _n.get("entity_id") or _n.get("id") or "")
        if _eid and "-" in _eid:
            try:
                _prefix, _num_str = _eid.rsplit("-", 1)
                _layer_max[_prefix] = max(_layer_max[_prefix], int(_num_str))
            except (ValueError, IndexError):
                pass
    return _layer_max

def _collect_fragment_ids(normalized: dict[str, Any]) -> set[str]:
    """Collect ALL fragment node IDs (nodes + instances) for intra-fragment collision avoidance."""
    _frag_original_ids: set[str] = set()
    for _n in normalized["nodes"] + normalized.get("instances", []):
        if isinstance(_n, dict):
            _eid = str(_n.get("meta_id") or _n.get("entity_id") or _n.get("id") or "")
            if _eid:
                _frag_original_ids.add(_eid)
    return _frag_original_ids

def _compute_id_remap(
    normalized: dict[str, Any],
    existing_id_set: set[str],
    layer_max: dict[str, int],
    frag_original_ids: set[str],
) -> tuple[dict[str, str], int]:
    """Detect fragment IDs colliding with the master and remap them to unique IDs."""
    _frag_id_remap: dict[str, str] = {}
    _reassigned_count = 0
    for _n in normalized["nodes"] + normalized.get("instances", []):
        if not isinstance(_n, dict):
            continue
        _eid = str(_n.get("meta_id") or _n.get("entity_id") or _n.get("id") or "")
        if not _eid or _eid not in existing_id_set:
            continue
        # Collision detected — generate new unique ID
        _prefix = _eid.rsplit("-", 1)[0] if "-" in _eid else _eid
        layer_max[_prefix] += 1
        _new_id = f"{_prefix}-{layer_max[_prefix]:03d}"
        # Ensure the new ID doesn't collide with: existing master IDs,
        # already-reassigned fragment IDs, or other fragment nodes' original IDs
        while (_new_id in existing_id_set
               or _new_id in _frag_id_remap.values()
               or _new_id in frag_original_ids):
            layer_max[_prefix] += 1
            _new_id = f"{_prefix}-{layer_max[_prefix]:03d}"
        _frag_id_remap[_eid] = _new_id
        if "meta_id" in _n:
            _n["meta_id"] = _new_id
        elif "entity_id" in _n:
            _n["entity_id"] = _new_id
        elif "id" in _n:
            _n["id"] = _new_id
        existing_id_set.add(_new_id)
        frag_original_ids.add(_new_id)
        _reassigned_count += 1
    return _frag_id_remap, _reassigned_count

def _apply_id_remap(normalized: dict[str, Any], frag_id_remap: dict[str, str]) -> None:
    """Point fragment relationships and parent_id references at remapped IDs."""
    # Update fragment relationships to use new IDs
    for _rel in normalized["relationships"]:
        if not isinstance(_rel, dict):
            continue
        for _key in ("source", "from", "from_id", "target", "to", "to_id"):
            _val = str(_rel.get(_key) or "")
            if _val in frag_id_remap:
                _rel[_key] = frag_id_remap[_val]

    # Update parent_id references in fragment nodes
    for _n in normalized["nodes"] + normalized.get("instances", []):
        if not isinstance(_n, dict):
            continue
        _pid = str(_n.get("parent_id") or "")
        if _pid in frag_id_remap:
            _n["parent_id"] = frag_id_remap[_pid]

def _reassign_colliding_ids(
    existing_nodes: list[dict[str, Any]],
    existing_instances: list[dict[str, Any]],
    normalized: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Reassign fragment IDs that collide with the existing master; update refs."""
    # The extract stage generates IDs starting from L1-001, L2-001, etc. for each run.
    # When merging with an existing master, these IDs collide with entities from previous
    # runs. This fix reassigns colliding fragment IDs to new unique IDs before merging.
    existing_id_set = _collect_existing_ids(existing_nodes, existing_instances)
    _reassigned_count = 0
    if existing_id_set and normalized["nodes"]:
        layer_max = _compute_layer_max(existing_nodes, existing_instances, normalized)
        frag_original_ids = _collect_fragment_ids(normalized)
        frag_id_remap, _reassigned_count = _compute_id_remap(
            normalized, existing_id_set, layer_max, frag_original_ids
        )
        _apply_id_remap(normalized, frag_id_remap)

        if _reassigned_count > 0:
            print(f"[Merge] ID collision fix: reassigned {_reassigned_count} fragment IDs to avoid collisions with existing master")
            for _old, _new in sorted(frag_id_remap.items()):
                print(f"  {_old} → {_new}")

    return normalized, _reassigned_count

def _migrate_l3_entities(
    all_nodes: list[dict[str, Any]],
    all_instances: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Move L3 entities out of nodes into instances, and migrate L1 system instances to L3."""
    def _is_l3_entity(node):
        if not isinstance(node, dict):
            return False
        if node.get("layer") == "L3":
            return True
        entity_id = node.get("entity_id", node.get("meta_id", ""))
        return entity_id.startswith("L3")

    l3_nodes_to_remove = []
    for i, node in enumerate(all_nodes):
        if _is_l3_entity(node):
            all_instances.append(node)
            l3_nodes_to_remove.append(i)

    for i in reversed(l3_nodes_to_remove):
        all_nodes.pop(i)

    # [L1->L3 MIGRATION]: Move specific system instances from L1 to L3
    l1_to_l3 = []
    for node in all_nodes:
        if node.get("layer") == "L1":
            name = str(node.get("name", "")).strip()
            is_instance = False
            # Check patterns
            for pattern in INSTANCE_PATTERNS:
                if pattern.match(name):
                    is_instance = True
                    break
            # Check known instances
            if name in KNOWN_INSTANCES:
                is_instance = True
            # Check meta_class=System with very specific name (has digits or known acronym)
            if node.get("meta_class") == "System" and re.search(r'\d', name):
                is_instance = True

            if is_instance:
                node["layer"] = "L3"
                # Update meta_id from L1-xxx to L3-xxx
                old_id = node.get("meta_id", "")
                if old_id.startswith("L1-"):
                    node["meta_id"] = old_id.replace("L1-", "L3-", 1)
                l1_to_l3.append(node)
                print(f"[Merge] L1->L3: '{name}' moved to L3 as system instance")

    # Remove moved nodes from all_nodes and add to all_instances
    if l1_to_l3:
        l1_to_l3_ids = {id(n) for n in l1_to_l3}
        all_nodes = [n for n in all_nodes if id(n) not in l1_to_l3_ids]
        all_instances.extend(l1_to_l3)
        print(f"[Merge] Moved {len(l1_to_l3)} entities from L1 to L3")

    return all_nodes, all_instances

def _layer_dedup(
    all_nodes: list[dict[str, Any]],
    all_instances: list[dict[str, Any]],
    dedup: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate nodes per layer (L0/L1/L2 custom, other by meta_id) and normalize."""
    l0_nodes = [n for n in all_nodes if n.get("layer") == "L0"]
    l1_nodes = [n for n in all_nodes if n.get("layer") == "L1"]
    l2_nodes = [n for n in all_nodes if n.get("layer") in ("L2", "L2.5")]
    other_nodes = [n for n in all_nodes if n.get("layer") not in ("L0", "L1", "L2", "L2.5")]

    def l0_key_extractor(node: dict) -> str:
        # L0 is a fixed abstract vocabulary. Keep it source-insensitive so the
        # auto-supplemented base classes do not duplicate across files.
        name = str(node.get("name") or node.get("meta_name") or "").strip().lower()
        meta_class = str(node.get("meta_class") or node.get("entity_type") or "").strip().lower()
        return f"{name}|{meta_class}"

    def l1_key_extractor(node: dict) -> str:
        # MBOM / manufacturing documents frequently reuse the same L1 label
        # across different DU / variant / position contexts. Keep provenance in
        # the signature so semantically distinct nodes are not collapsed.
        return "|".join(_entity_signature(node, include_source=True))

    def l2_key_extractor(node: dict) -> str:
        return "|".join(_entity_signature(node, include_source=True))

    l0_merged = dedup.deduplicate_nodes(l0_nodes, by="custom", key_extractor=l0_key_extractor)
    l1_merged = dedup.deduplicate_nodes(l1_nodes, by="custom", key_extractor=l1_key_extractor)
    l2_merged = dedup.deduplicate_nodes(l2_nodes, by="custom", key_extractor=l2_key_extractor)
    other_merged = dedup.deduplicate_nodes(other_nodes, by="meta_id")

    merged_nodes = l0_merged + l1_merged + l2_merged + other_merged
    merged_instances = dedup.deduplicate_nodes(all_instances, by="meta_id")

    merged_nodes, merged_instances = _normalize_all_entities(merged_nodes, merged_instances)

    # Preserve MBOM-specific entities even if normalization causes naming collapse.
    # This keeps Delivery Unit / Position / Variant / Occurrence visible in the final master.
    for node in merged_nodes:
        if not isinstance(node, dict):
            continue
        nname = str(node.get("name") or node.get("meta_name") or "").strip().lower()
        if any(token in nname for token in MBOM_KEEP_NAMES):
            node.setdefault("mbom_signal", True)

    return merged_nodes, merged_instances

def _build_merged_nodes(
    existing_nodes: list[dict[str, Any]],
    existing_instances: list[dict[str, Any]],
    normalized: dict[str, Any],
    dedup: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge existing + fragment entities/instances with layer-aware dedup."""
    all_nodes = existing_nodes + normalized["nodes"]
    all_instances = existing_instances + normalized["instances"]
    all_nodes, all_instances = _migrate_l3_entities(all_nodes, all_instances)
    return _layer_dedup(all_nodes, all_instances, dedup)

def _collect_entity_ids(
    merged_nodes: list[dict[str, Any]],
    merged_instances: list[dict[str, Any]],
) -> set[str]:
    """Collect all entity/instance IDs for dangling-relationship detection."""
    node_ids: set[str] = set()
    for n in merged_nodes:
        eid = str(n.get("meta_id") or n.get("entity_id") or n.get("id") or "")
        if eid:
            node_ids.add(eid)
    for n in merged_instances:
        eid = str(n.get("meta_id") or n.get("entity_id") or n.get("id") or "")
        if eid:
            node_ids.add(eid)
    return node_ids

def _collect_entity_name_index(
    merged_nodes: list[dict[str, Any]],
    merged_instances: list[dict[str, Any]],
    priority_nodes: list[dict[str, Any]] | None = None,
    priority_instances: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Map lowercased entity names to their meta_id for name→id endpoint fallback.

    Fragment-injected entities (priority_nodes/priority_instances) are indexed
    first so their IDs win over older same-named master nodes. A fresh
    fragment's name-based relations (e.g. safety-net EBOM→BOM) must resolve
    to this run's entities, not a stale master node that merely shares the
    label (e.g. master L1 node with scenario_name="MBOM").
    """
    index: dict[str, str] = {}

    def _add(nodes: list[dict[str, Any]]) -> None:
        for n in nodes:
            if not isinstance(n, dict):
                continue
            eid = str(n.get("meta_id") or n.get("entity_id") or n.get("id") or "")
            if not eid:
                continue
            for key in ("name", "meta_name", "scenario_name"):
                v = n.get(key)
                if isinstance(v, str) and v.strip():
                    index.setdefault(v.strip().lower(), eid)

    _add((priority_nodes or []) + (priority_instances or []))
    _add(merged_nodes + merged_instances)
    return index

def _resolve_rel_endpoints(
    rel: dict[str, Any],
    name_index: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve a relationship's source/target to plain IDs (dict endpoints unwrapped).

    When ``name_index`` (lowercased entity name -> meta_id) is provided, name-based
    endpoints (e.g. safety-net relations carrying ``source``/``target`` as names)
    are resolved to entity IDs so the ID-space dangling check can validate them.
    """
    source = rel.get("source") or rel.get("from")
    target = rel.get("target") or rel.get("to")

    if isinstance(source, dict):
        source_id = str(source.get("meta_id") or source.get("entity_id") or source.get("id") or "")
    else:
        source_id = str(source or "")

    if isinstance(target, dict):
        target_id = str(target.get("meta_id") or target.get("entity_id") or target.get("id") or "")
    else:
        target_id = str(target or "")

    if name_index:
        if source_id:
            source_id = name_index.get(source_id.lower(), source_id)
        if target_id:
            target_id = name_index.get(target_id.lower(), target_id)

    return source_id, target_id

def _normalize_relationship(
    rel: dict[str, Any],
    name_index: dict[str, str] | None = None,
) -> dict[str, Any]:
    source_id, target_id = _resolve_rel_endpoints(rel, name_index)
    rel["source"] = source_id
    rel["target"] = target_id
    if "from" in rel:
        rel.pop("from")
    if "to" in rel:
        rel.pop("to")

    confidence = rel.get("confidence")
    if confidence is None or confidence == 0 or confidence == "":
        properties = rel.get("properties", {})
        if isinstance(properties, dict):
            prop_conf = properties.get("confidence")
            if prop_conf:
                if isinstance(prop_conf, str):
                    conf_map = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}
                    confidence = conf_map.get(prop_conf.upper(), 0.7)
                else:
                    confidence = float(prop_conf)

        if confidence is None or confidence == 0:
            confidence = 0.7

    rel["confidence"] = float(confidence)
    rel.setdefault("origin_kind", "derived")
    if rel.get("run_id"):
        rel.setdefault("origin_run_id", rel["run_id"])

    rel_type = rel.get("relation_type") or rel.get("type") or rel.get("rel_type") or ""
    if rel_type:
        rel_type = str(rel_type).upper().strip()
        rel["relation_type"] = rel_type
        rel["type"] = rel_type
    for k in list(rel.keys()):
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", k):
            del rel[k]
    return rel

def _set_origin(item: Dict[str, Any], kind: str, run_id: str = "") -> None:
    if isinstance(item, dict):
        item.setdefault("origin_kind", kind)
        if run_id:
            item.setdefault("origin_run_id", run_id)

def _mark_origins(data: Dict[str, Any], kind: str, run_id: str = "") -> Dict[str, Any]:
    keys = ("nodes", "instances", "relationships", "edges")
    for key in keys:
        for item in data.get(key) or []:
            if isinstance(item, dict):
                item.setdefault("origin_kind", kind)
                if run_id:
                    item.setdefault("origin_run_id", run_id)
    return data

def _mark_master_origins(master: Dict[str, Any], run_id: str = "") -> Dict[str, Any]:
    return _mark_origins(master, "master_reused", run_id)

def _mark_normalized_origins(normalized: Dict[str, Any], run_id: str = "") -> Dict[str, Any]:
    return _mark_origins(normalized, "input", run_id)

def _merge_relationships(
    master: dict[str, Any],
    normalized: dict[str, Any],
    merged_nodes: list[dict[str, Any]],
    merged_instances: list[dict[str, Any]],
    dedup: Any,
) -> tuple[dict[str, Any], int]:
    relationships = (master.get("relationships") or master.get("edges") or []) + normalized["relationships"]
    deduplicated_relationships = dedup.deduplicate_edges(relationships)
    node_ids = _collect_entity_ids(merged_nodes, merged_instances)
    name_index = _collect_entity_name_index(
        merged_nodes,
        merged_instances,
        priority_nodes=normalized["nodes"],
        priority_instances=normalized["instances"],
    )

    valid_relationships = []
    dangling_count = 0
    for rel in deduplicated_relationships:
        if not isinstance(rel, dict):
            continue
        source_id, target_id = _resolve_rel_endpoints(rel, name_index)
        if source_id in node_ids and target_id in node_ids:
            valid_relationships.append(_normalize_relationship(rel, name_index))
        else:
            dangling_count += 1

    master["relationships"] = valid_relationships
    return master, dangling_count

def _deduplicate_list(items: list, id_keys: list | None = None, require_id: bool = False) -> tuple[list, int]:
    if id_keys is None:
        id_keys = ["meta_id", "id", "entity_id", "name"]
    seen = set()
    deduped = []
    removed = 0
    for item in items:
        if not isinstance(item, dict):
            removed += 1
            continue
        item_id_parts = []
        for key in id_keys:
            if key in item:
                val = str(item[key])
                if val and val != 'N/A' and val != 'None':
                    item_id_parts.append(val)
        item_id = '_'.join(item_id_parts)
        if require_id and not item_id:
            removed += 1
            continue
        if item_id and item_id in seen:
            removed += 1
            continue
        if item_id:
            seen.add(item_id)
        if isinstance(item, dict):
            item = _keep_entity_provenance(item, item)
        deduped.append(item)
    return deduped, removed

def _populate_layer_views(master: dict[str, Any], merged_nodes: list[dict[str, Any]], merged_instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Populate master entities/legacy layer views and deduplicate legacy lists."""
    master.setdefault("entities", {})
    layer_map: dict[str, list[str]] = {
        "L0": [], "L1": [], "L2": [], "L2.5": [], "L3": [],
    }
    legacy_map: dict[str, str] = {
        "L0": "layer0_metadata",
        "L1": "layer1_concepts",
        "L2": "layer2_scenarios",
        "L2.5": "layer25_mapping_rules",
    }
    for n in merged_nodes:
        layer = _entity_layer_value(n)
        if layer in layer_map:
            layer_map[layer].append(n)
        legacy_key = legacy_map.get(layer)
        if legacy_key:
            master.setdefault(legacy_key, [])
            if n not in master[legacy_key]:
                master[legacy_key].append(n)
    for layer_key, bucket in layer_map.items():
        master["entities"][layer_key] = bucket
    for legacy_key in ("layer2_definitions", "layer2_logic", "layer25_mapping_rules"):
        if legacy_key not in master:
            master[legacy_key] = []
    master["entities"]["L3"] = merged_instances

    layer_id_keys = {
        "layer0_metadata": ["meta_id", "id", "name", "file_path", "section"],
        "layer1_concepts": ["meta_id", "id", "entity_id", "name"],
        "layer2_scenarios": ["meta_id", "id", "scenario_name", "name", "description", "source_file"],
        "layer25_mapping_rules": ["meta_id", "id", "name", "source_concept", "target_concept", "rule_text"],
    }
    layer_require_id = {
        "layer0_metadata": False,
        "layer1_concepts": False,
        "layer2_scenarios": False,
        "layer25_mapping_rules": True,
    }

    deduplicated_layers = {}
    for legacy_key in ["layer0_metadata", "layer1_concepts", "layer2_scenarios", "layer25_mapping_rules"]:
        if legacy_key in master and isinstance(master[legacy_key], list):
            original_count = len(master[legacy_key])
            keys = layer_id_keys.get(legacy_key, ["meta_id", "id", "entity_id", "name"])
            require_id = layer_require_id.get(legacy_key, False)
            master[legacy_key], removed = _deduplicate_list(master[legacy_key], keys, require_id)
            deduplicated_layers[legacy_key] = {
                "original": original_count,
                "deduped": len(master[legacy_key]),
                "removed": removed,
            }
    return deduplicated_layers

def _write_master(master_path: Path, master: dict[str, Any]) -> dict[str, Any]:
    """Persist master to store (or JSON fallback); returns compaction results."""
    if MasterOntologyStore:
        store = create_master_store(str(master_path), sync_mode=True)
        store.write(master)
        compact_results = store.compact()
    else:
        _write_json(master_path, master)
        compact_results = {}
    return compact_results


def _fragment_entity_count(normalized: dict[str, Any]) -> int:
    """Count extractable entities carried by a normalized fragment."""
    return (
        len(normalized.get("nodes", []))
        + len(normalized.get("instances", []))
        + len(normalized.get("relationships", []))
    )


def merge_min_fragment_entities() -> int:
    """Minimum entity count a fragment must carry for merge to accept it.

    Powers the empty-extraction guard (P0-3): an empty or under-threshold
    extract must never be merged into master, otherwise an unstable LLM
    provider can silently wipe master (regression 2026-09-01, GOV rollback).
    Defaults to 0, which blocks only absolute-empty extractions; raise the
    bar with the MERGE_MIN_FRAGMENT_ENTITIES env var.
    """
    raw = os.environ.get("MERGE_MIN_FRAGMENT_ENTITIES", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def run_merge(args: Namespace) -> StageResult:
    start = time.time()
    from ontology_compiler.services.merge.entity_dedup_service import EntityDeduplicationService

    fragment_path, master_path, run_dir = _resolve_merge_inputs(args)
    if not fragment_path or not fragment_path.exists():
        return StageResult("merge", False, "no fragment artifact found; run extract first", elapsed_seconds=round(time.time() - start, 3))

    master = _load_master(master_path)
    _mark_master_origins(master, getattr(args, "run_id", ""))
    _fragment, normalized = _load_fragment(fragment_path)
    _mark_normalized_origins(normalized, getattr(args, "run_id", ""))

    # Empty-extraction guard (P0-3): never write master from an empty or
    # under-threshold extract. An unstable LLM provider can silently return an
    # empty fragment which would otherwise wipe the whole master.
    _min_entities = merge_min_fragment_entities()
    _fragment_count = _fragment_entity_count(normalized)
    if _fragment_count == 0 or _fragment_count < _min_entities:
        return StageResult(
            "merge",
            False,
            f"empty/under-threshold extraction blocked: fragment '{fragment_path.name}' "
            f"carries {_fragment_count} entities "
            f"(min={_min_entities}); refusing to write master",
            elapsed_seconds=round(time.time() - start, 3),
        )

    existing_nodes = _collect_legacy_nodes(master, master.get("nodes", []))
    existing_instances = master.get("instances", [])

    normalized, _ = _reassign_colliding_ids(existing_nodes, existing_instances, normalized)

    dedup = EntityDeduplicationService(strategy="fingerprint")
    merged_nodes, merged_instances = _build_merged_nodes(existing_nodes, existing_instances, normalized, dedup)

    master["nodes"] = merged_nodes
    master["instances"] = merged_instances

    try:
        from src.pipeline.stages.extract import _apply_node_optimizations
        master["nodes"] = _apply_node_optimizations(master.get("nodes", []))
        print(f"[Merge] Applied node optimizations to {len(master.get('nodes', []))} nodes")
    except Exception as e:
        print(f"[Merge] Node optimization skipped: {e}")

    master, dangling_count = _merge_relationships(master, normalized, merged_nodes, merged_instances, dedup)
    deduplicated_layers = _populate_layer_views(master, merged_nodes, merged_instances)

    global_opt_stats: dict[str, Any] = {}
    try:
        from .global_optimizer import optimize_before_master_write
        master, global_opt_stats = optimize_before_master_write(master, getattr(args, "run_id", ""))
        print(f"[Merge] Global optimization: {global_opt_stats}")
    except Exception as e:
        print(f"[Merge] Global optimization skipped: {e}")

    compact_results = _write_master(master_path, master)

    new_nodes = len(merged_nodes) - len(existing_nodes)
    duplicate_nodes = len(existing_nodes) + len(normalized["nodes"]) - len(merged_nodes)

    return StageResult(
        "merge",
        True,
        "merge completed",
        {
            "merged_nodes": len(merged_nodes),
            "merged_instances": len(merged_instances),
            "merged_relationships": len(master["relationships"]),
            "new_nodes": new_nodes,
            "duplicate_nodes": duplicate_nodes,
            "removed_dangling_relationships": dangling_count,
            "deduplicated_layers": deduplicated_layers,
            "compacted_layers": compact_results,
            "master_path": str(master_path),
            "is_directory_store": MasterOntologyStore is not None,
            "global_optimization": global_opt_stats,
        },
        round(time.time() - start, 3),
        artifact_paths=[str(fragment_path), str(master_path)],
        run_dir=str(run_dir) if run_dir else "",
        fragment_path=str(fragment_path),
    )

def _prepare_dedup_artifacts(master_path: Path, args: Namespace) -> tuple[Path, Path, dict[str, Any], bool]:
    """Create backup + audit paths and backup metadata for a dedup run."""
    backup_path = master_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    shutil.copy2(master_path, backup_path)
    dry_run = bool(getattr(args, "dry_run", False))
    audit_path = master_path.with_suffix(f".dedup_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    backup_metadata = build_backup_metadata(str(master_path), str(backup_path), str(getattr(args, "run_id", "")), "dedup")
    return backup_path, audit_path, backup_metadata, dry_run

def _deduplicate_master(master: dict[str, Any], dedup: Any) -> dict[str, list[dict[str, Any]]]:
    """Deduplicate master nodes/instances/relationships by custom signature."""
    nodes = master.get("nodes", [])
    instances = master.get("instances", [])
    relationships = master.get("relationships") or master.get("edges") or []
    deduped_nodes = dedup.deduplicate_nodes(
        nodes, by="custom", key_extractor=lambda n: "|".join(_entity_signature(n, include_source=True))
    )
    deduped_instances = dedup.deduplicate_nodes(
        instances, by="custom", key_extractor=lambda n: "|".join(_entity_signature(n, include_source=True))
    )
    deduped_relationships = dedup.deduplicate_edges(relationships)
    deduped_nodes, deduped_instances = _normalize_all_entities(deduped_nodes, deduped_instances)
    return {
        "nodes": nodes,
        "instances": instances,
        "relationships": relationships,
        "deduped_nodes": deduped_nodes,
        "deduped_instances": deduped_instances,
        "deduped_relationships": deduped_relationships,
    }

def _build_dedup_audit(
    dry_run: bool,
    backup_metadata: dict[str, Any],
    backup_path: Path,
    audit_path: Path,
    master_path: Path,
    result: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build the dedup audit report (before/after counts and payloads)."""
    nodes, instances, relationships = result["nodes"], result["instances"], result["relationships"]
    deduped_nodes, deduped_instances, deduped_relationships = (
        result["deduped_nodes"], result["deduped_instances"], result["deduped_relationships"],
    )
    return {
        "timestamp": datetime.now().isoformat(),
        "mode": "dry_run" if dry_run else "commit",
        "thresholds": PIPELINE_VALIDATION_THRESHOLDS,
        "backup": backup_metadata,
        "backup_path": str(backup_path),
        "master_path": str(master_path),
        "summary": {
            "original_nodes": len(nodes),
            "deduped_nodes": len(deduped_nodes),
            "removed_nodes": len(nodes) - len(deduped_nodes),
            "original_instances": len(instances),
            "deduped_instances": len(deduped_instances),
            "removed_instances": len(instances) - len(deduped_instances),
            "original_relationships": len(relationships),
            "deduped_relationships": len(deduped_relationships),
            "removed_relationships": len(relationships) - len(deduped_relationships),
        },
        "node_changes": {"before": nodes, "after": deduped_nodes},
        "instance_changes": {"before": instances, "after": deduped_instances},
        "relationship_changes": {"before": relationships, "after": deduped_relationships},
    }

def _apply_dedup_to_master(
    master: dict[str, Any],
    deduped_nodes: list[dict[str, Any]],
    deduped_instances: list[dict[str, Any]],
    deduped_relationships: list[dict[str, Any]],
    backup_metadata: dict[str, Any],
    audit_path: Path,
) -> dict[str, Any]:
    """Overwrite master collections with deduped results and record metadata."""
    master["nodes"] = deduped_nodes
    master["instances"] = deduped_instances
    master["relationships"] = deduped_relationships
    master.setdefault("metadata", {})
    master["metadata"]["dedup_backup"] = backup_metadata
    master["metadata"]["dedup_audit_path"] = str(audit_path)
    return master

def run_dedup(args: Namespace) -> StageResult:
    start = time.time()
    from ontology_compiler.services.merge.entity_dedup_service import EntityDeduplicationService

    master_path = Path(args.master_path)
    if not master_path.exists():
        return StageResult("dedup", False, f"master not found: {master_path}", elapsed_seconds=round(time.time() - start, 3))

    master = _read_json(master_path, {})
    backup_path, audit_path, backup_metadata, dry_run = _prepare_dedup_artifacts(master_path, args)

    dedup = EntityDeduplicationService(strategy="fingerprint")
    result = _deduplicate_master(master, dedup)

    audit_report = _build_dedup_audit(dry_run, backup_metadata, backup_path, audit_path, master_path, result)
    _write_json(audit_path, audit_report)

    if not dry_run:
        master = _apply_dedup_to_master(
            master,
            result["deduped_nodes"],
            result["deduped_instances"],
            result["deduped_relationships"],
            backup_metadata,
            audit_path,
        )
        _write_json(master_path, master)

    return StageResult(
        "dedup",
        True,
        "dedup dry-run completed" if dry_run else "dedup completed",
        {
            **audit_report["summary"],
            "backup_path": str(backup_path),
            "audit_path": str(audit_path),
            "dry_run": dry_run,
        },
        round(time.time() - start, 3),
        artifact_paths=[str(master_path), str(backup_path), str(audit_path)],
        run_dir="",
        fragment_path="",
    )
