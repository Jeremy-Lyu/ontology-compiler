"""Extraction pipeline stages for the unified extractor.

Stage functions extracted from the inline body of ``extract_async_content`` in
``true_unified_extractor`` so the orchestrator reads as a sequence of named
steps. Each stage mutates the shared ``nodes`` list and the shared
``counter`` / ``id_mapping`` state exactly as the original inline code did.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List

from ontology_compiler.services.extraction.extraction_postprocessing import (
    correct_meta_class,
    GENERIC_L1_TERMS,
    GENERIC_L2_TERMS,
    L1_ALLOWED_ENUMERATION,
    L1_ORG_KEYWORDS,
    L1_TOO_SPECIFIC_PATTERNS,
    L2_TO_L3_PATTERNS,
    PART_INSTANCE_PATTERNS,
    is_noise_entity,
    normalize_evidence,
    normalize_entity_type,
    normalize_relationship,
    resolve_entity_id,
)

logger = logging.getLogger(__name__)


# --- L1->L2 re-classification -------------------------------------------------

# Keywords that indicate CONCRETE implementations (move to L2)
L1_TO_L2_KEYWORDS = {
    "chimas", "vida", "vcats", "ecu", "vpds", "teamcenter", "sap",
    "mes", "kdp", "cmp", "plus", "pif", "mpp", "ohf", "aro",
    "hercules", "qnx", "vteststudio", "ipca", "codev", "pie",
    "sharepoint", "powerbi", "dms", "kompass", "tacdis", "crlog",
    "vccbiz", "cdb", "siplus", "qaenvironment", "cpr780", "si",
    "aftersales", "production", "warehouse", "logistics",
    "assembly", "assembly line", "body shop", "paint shop", "plant",
    "goods reception", "kanban", "cycle counting", "replenishment",
    "software download", "fault tracing", "change order",
    "diagnostics", "parts management", "service operations",
    "order handling", "order handling system",
}

# Patterns that indicate ABSTRACT subtypes (KEEP in L1)
L1_RETENTION_PATTERNS = [
    r".*[Bb][Oo][Mm]$",
    r"^[Pp]rocess$",
    r"^[Ss]ystem$",
    r"^[Cc]apability$",
    r"^[Aa]sset$",
    r"^[Cc]oncept$",
    r"^[Aa]ctor$",
    r"^SupplyChain$",
    r"^Manufacturing$",
    r"^Quality$",
    r"^Engineering$",
]


def reclassify_l1_to_l2(
    nodes: List[Dict],
    seen_ids: dict,
    counter: dict,
    id_mapping: dict,
    l0_class_to_id: dict,
    valid_l0_classes: set,
) -> None:
    """Move concrete implementations from L1 to L2 (white-list approach)."""
    l1_to_l2_mapping: Dict[str, str] = {}

    l1_entities_before = [node.get("name", "") for node in nodes if node.get("layer") == "L1"]
    print(f"[LAYER RE-CLASSIFY DEBUG] L1 entities before reclassification: {l1_entities_before}", flush=True)
    logger.info(f"[LAYER RE-CLASSIFY DEBUG] L1 entities before reclassification: {l1_entities_before}")

    for node in nodes:
        if node.get("layer") == "L1":
            name = node.get("name", "").strip()
            name_lower = name.lower()

            print(f"[LAYER RE-CLASSIFY DEBUG] Checking L1 node: name='{name}', name_lower='{name_lower}', meta_id='{node.get('meta_id', '')}'", flush=True)

            should_stay_in_l1 = False
            for pattern in L1_RETENTION_PATTERNS:
                if re.match(pattern, name):
                    should_stay_in_l1 = True
                    print(f"[LAYER RE-CLASSIFY DEBUG] '{name}' matched retention pattern '{pattern}', staying in L1", flush=True)
                    break

            if should_stay_in_l1:
                continue

            is_concrete = False

            matched_keyword = None
            for keyword in L1_TO_L2_KEYWORDS:
                if re.search(r'\b' + re.escape(keyword) + r'\b', name_lower):
                    matched_keyword = keyword
                    is_concrete = True
                    break

            if matched_keyword:
                print(f"[LAYER RE-CLASSIFY DEBUG] '{name}' matched keyword rule: '{matched_keyword}'", flush=True)
                logger.info(f"[LAYER RE-CLASSIFY DEBUG] '{name}' matched keyword rule")
            elif re.match(r"^[A-Z_]{2,}$", name):
                is_concrete = True
                print(f"[LAYER RE-CLASSIFY DEBUG] '{name}' matched acronym rule", flush=True)
                logger.info(f"[LAYER RE-CLASSIFY DEBUG] '{name}' matched acronym rule")
            elif re.match(r"^[A-Z]+\d+", name):
                is_concrete = True
                print(f"[LAYER RE-CLASSIFY DEBUG] '{name}' matched acronym-with-numbers rule", flush=True)
                logger.info(f"[LAYER RE-CLASSIFY DEBUG] '{name}' matched acronym-with-numbers rule")
            elif re.match(r"^[A-Z]+[a-z]+[A-Z]", name):
                is_concrete = True
                print(f"[LAYER RE-CLASSIFY DEBUG] '{name}' matched CamelCase acronym rule", flush=True)
                logger.info(f"[LAYER RE-CLASSIFY DEBUG] '{name}' matched CamelCase acronym rule")

            if is_concrete:
                old_id = node["meta_id"]
                new_id = f"L2-{counter['L2']:03d}"
                counter["L2"] += 1

                node["layer"] = "L2"
                node["meta_id"] = new_id
                l1_to_l2_mapping[old_id] = new_id

                if "parent_class" in node and node["parent_class"] in valid_l0_classes:
                    l1_intermediate_name = f"{node['parent_class']}"
                    l1_key = ("L1", l1_intermediate_name, node["parent_class"])
                    if l1_key not in seen_ids:
                        l1_inter_id = f"L1-{counter['L1']:03d}"
                        counter["L1"] += 1
                        l1_intermediate = {
                            "meta_id": l1_inter_id,
                            "name": l1_intermediate_name,
                            "meta_class": node["parent_class"],
                            "layer": "L1",
                            "parent_class": node["parent_class"],
                            "parent_id": l0_class_to_id.get(node["parent_class"].lower(), ""),
                            "description": f"Abstract {node['parent_class']} subtype",
                            "source_file": node.get("source_file", ""),
                            "attributes": [],
                            "evidence": {"source_text": "Auto-generated intermediate"},
                        }
                        seen_ids[l1_key] = l1_intermediate
                        nodes.append(l1_intermediate)
                    else:
                        l1_inter_id = seen_ids[l1_key]["meta_id"]
                    node["parent_id"] = l1_inter_id

                logger.info(f"[LAYER RE-CLASSIFY] Moved '{name_lower}' from L1 to L2 (ID: {old_id} -> {new_id})")

    if l1_to_l2_mapping:
        id_mapping.update(l1_to_l2_mapping)


def reclassify_l2_to_l1(nodes: List[Dict], counter: dict, id_mapping: dict) -> None:
    """Move abstract subtypes from L2 back to L1."""
    l2_to_l1_mapping: Dict[str, str] = {}
    l2_entities_before = [n.get("name", "") for n in nodes if n.get("layer") == "L2"]
    print(f"[L2->L1 REVERSE DEBUG] Checking {len(l2_entities_before)} L2 entities for abstract patterns", flush=True)

    for node in nodes:
        if node.get("layer") == "L2":
            name = node.get("name", "").strip()
            meta_class = node.get("meta_class", "").strip()

            is_abstract = False

            if meta_class == "Capability":
                if name.endswith("Management") or name.endswith("Governance"):
                    is_abstract = True

            if not is_abstract and meta_class == "Process":
                if name.endswith("Management") and len(name) > len("Management") + 2:
                    is_abstract = True

            if not is_abstract and meta_class == "Data":
                if re.match(r"^[A-Z][a-zA-Z]*Data$", name):
                    is_abstract = True

            if is_abstract:
                old_id = node["meta_id"]
                new_id = f"L1-{counter['L1']:03d}"
                counter["L1"] += 1

                node["layer"] = "L1"
                node["meta_id"] = new_id
                l2_to_l1_mapping[old_id] = new_id

                if "parent_class" not in node:
                    node["parent_class"] = node.get("meta_class", "Capability")

                print(f"[L2->L1 REVERSE] Moved '{name}' from L2 to L1 (ID: {old_id} -> {new_id})", flush=True)
                logger.info(f"[L2->L1 REVERSE] Moved '{name}' from L2 to L1 (ID: {old_id} -> {new_id})")

    if l2_to_l1_mapping:
        id_mapping.update(l2_to_l1_mapping)


# --- Entity quality filter v2 --------------------------------------------------

def apply_layer_quality_filter(
    nodes: List[Dict],
    counter: dict,
    id_mapping: dict,
    relationships: List[Dict],
) -> List[Dict]:
    """Enforce strict layer constraints, disambiguate L2 names, merge L1 duplicates.

    Returns the filtered ``nodes`` list (original list is not reused after the
    removal step) and mutates ``relationships`` in place.
    """
    entities_to_remove: set = set()

    for node in nodes:
        layer = node.get("layer", "")
        name = node.get("name", "").strip()
        name_lower = name.lower()
        meta_class = node.get("meta_class", "").strip()
        parent_class = node.get("parent_class", "").strip()
        meta_id = node.get("meta_id", "")

        if layer == "L1":
            if name_lower == parent_class.lower():
                entities_to_remove.add(meta_id)
                print(f"[QUALITY FILTER] Removed L1 '{name}' (duplicates parent_class '{parent_class}')", flush=True)
                logger.info(f"[QUALITY FILTER] Removed L1 '{name}' (duplicates parent_class '{parent_class}')")
                continue

            if name_lower in GENERIC_L1_TERMS:
                entities_to_remove.add(meta_id)
                print(f"[QUALITY FILTER] Removed L1 '{name}' (generic term, not a domain subtype)", flush=True)
                logger.info(f"[QUALITY FILTER] Removed L1 '{name}' (generic term, not a domain subtype)")
                continue

            name_words = name_lower.split()
            if any(kw in name_words for kw in L1_ORG_KEYWORDS):
                old_id = node["meta_id"]
                new_id = f"L2-{counter['L2']:03d}"
                counter["L2"] += 1
                node["layer"] = "L2"
                node["meta_id"] = new_id
                id_mapping[old_id] = new_id
                if meta_class != "Actor":
                    node["meta_class"] = "Actor"
                print(f"[QUALITY FILTER] Moved '{name}' from L1 to L2 (organization/unit, not abstract: {old_id} -> {new_id})", flush=True)
                logger.info(f"[QUALITY FILTER] Moved '{name}' from L1 to L2 (organization/unit)")
                continue

            if meta_class == "Data" and name_lower not in L1_ALLOWED_ENUMERATION:
                if name_lower.endswith("data") and len(name) > 5:
                    old_id = node["meta_id"]
                    new_id = f"L2-{counter['L2']:03d}"
                    counter["L2"] += 1
                    node["layer"] = "L2"
                    node["meta_id"] = new_id
                    id_mapping[old_id] = new_id
                    print(f"[QUALITY FILTER] Moved '{name}' from L1 to L2 (specific data entity: {old_id} -> {new_id})", flush=True)
                    logger.info(f"[QUALITY FILTER] Moved '{name}' from L1 to L2 (specific data entity)")
                    continue

            is_too_specific = False
            for pattern in L1_TOO_SPECIFIC_PATTERNS:
                if re.match(pattern, name):
                    is_too_specific = True
                    break

            if not is_too_specific and re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", name):
                second_word = name.split()[-1].lower()
                abstract_suffixes = {"management", "governance", "strategy", "process", "system", "data"}
                if second_word not in abstract_suffixes:
                    is_too_specific = True

            if is_too_specific:
                old_id = node["meta_id"]
                new_id = f"L2-{counter['L2']:03d}"
                counter["L2"] += 1
                node["layer"] = "L2"
                node["meta_id"] = new_id
                id_mapping[old_id] = new_id
                print(f"[QUALITY FILTER] Moved '{name}' from L1 to L2 (too specific for L1: {old_id} -> {new_id})", flush=True)
                logger.info(f"[QUALITY FILTER] Moved '{name}' from L1 to L2 (too specific for L1)")
                continue

        if layer == "L2":
            if name_lower in GENERIC_L2_TERMS:
                entities_to_remove.add(meta_id)
                print(f"[QUALITY FILTER] Removed L2 '{name}' (generic term, not a concrete implementation)", flush=True)
                logger.info(f"[QUALITY FILTER] Removed L2 '{name}' (generic term, not a concrete implementation)")
                continue

            corrected_class = correct_meta_class(name, meta_class)
            if corrected_class != meta_class:
                print(f"[QUALITY FILTER] Corrected '{name}' meta_class: {meta_class} -> {corrected_class}", flush=True)
                logger.info(f"[QUALITY FILTER] Corrected '{name}' meta_class: {meta_class} -> {corrected_class}")
                node["meta_class"] = corrected_class
                if "parent_class" in node:
                    node["parent_class"] = corrected_class

            should_downgrade_to_l3 = False
            for pattern in L2_TO_L3_PATTERNS:
                if re.match(pattern, name):
                    should_downgrade_to_l3 = True
                    break

            if should_downgrade_to_l3:
                old_id = node["meta_id"]
                new_id = f"L3-{counter['L3']:03d}"
                counter["L3"] += 1
                node["layer"] = "L3"
                node["meta_id"] = new_id
                id_mapping[old_id] = new_id
                print(f"[QUALITY FILTER] Downgraded '{name}' from L2 to L3 (physical/micro instance: {old_id} -> {new_id})", flush=True)
                logger.info(f"[QUALITY FILTER] Downgraded '{name}' from L2 to L3 (physical/micro instance)")
                continue

            is_part_instance = False
            for pattern in PART_INSTANCE_PATTERNS:
                if re.match(pattern, name):
                    is_part_instance = True
                    break

            if is_part_instance and meta_class == "Asset":
                old_id = node["meta_id"]
                new_id = f"L3-{counter['L3']:03d}"
                counter["L3"] += 1
                node["layer"] = "L3"
                node["meta_id"] = new_id
                id_mapping[old_id] = new_id
                print(f"[QUALITY FILTER] Moved '{name}' from L2 to L3 (part instance: {old_id} -> {new_id})", flush=True)
                logger.info(f"[QUALITY FILTER] Moved '{name}' from L2 to L3 (part instance)")

    # L2 name disambiguation: same name with different meta_class gets type suffix
    l2_name_class_map: Dict[str, list] = {}
    for node in nodes:
        if node.get("layer") == "L2":
            name = node.get("name", "").strip()
            normalized = re.sub(r"\s+", "", name).lower()
            mc = node.get("meta_class", "")
            l2_name_class_map.setdefault(normalized, []).append((node, mc))

    for normalized, entries in l2_name_class_map.items():
        if len(entries) > 1:
            classes = set(mc for _, mc in entries)
            if len(classes) > 1:
                for node, mc in entries:
                    old_name = node["name"]
                    type_suffix = f"_{mc}"
                    if not old_name.endswith(type_suffix):
                        node["name"] = old_name + type_suffix
                        print(f"[QUALITY FILTER] Disambiguated L2 '{old_name}' -> '{node['name']}' (meta_class={mc})", flush=True)
                        logger.info(f"[QUALITY FILTER] Disambiguated L2 '{old_name}' -> '{node['name']}'")

    # L1 deduplication: merge entities with same normalized name
    l1_name_map: Dict[str, Dict] = {}
    l1_duplicates: set = set()
    for node in nodes:
        if node.get("layer") == "L1":
            name = node.get("name", "").strip()
            normalized = re.sub(r"\s+", "", name).lower()
            if normalized in l1_name_map:
                old_id = node["meta_id"]
                keep_id = l1_name_map[normalized]["meta_id"]
                l1_duplicates.add(old_id)
                id_mapping[old_id] = keep_id
                print(f"[QUALITY FILTER] Merged L1 duplicate '{name}' ({old_id} -> {keep_id})", flush=True)
                logger.info(f"[QUALITY FILTER] Merged L1 duplicate '{name}' ({old_id} -> {keep_id})")
            else:
                l1_name_map[normalized] = node

    if l1_duplicates:
        nodes = [n for n in nodes if n.get("meta_id") not in l1_duplicates]
        entities_to_remove.update(l1_duplicates)

    if entities_to_remove:
        nodes = [n for n in nodes if n.get("meta_id") not in entities_to_remove]
        if relationships:
            original_rel_count = len(relationships)
            relationships[:] = [
                r for r in relationships
                if r.get("from", r.get("from_id", r.get("source", ""))) not in entities_to_remove
                and r.get("to", r.get("to_id", r.get("target", ""))) not in entities_to_remove
            ]
            removed_rels = original_rel_count - len(relationships)
            print(f"[QUALITY FILTER] Removed {len(entities_to_remove)} entities and {removed_rels} associated relationships", flush=True)

    return nodes


def normalize_and_filter_relationships(
    result, deduplicated_nodes, deduplicated_instances, valid_node_ids,
    id_mapping, name_to_id_map, entity_map, entity_layer_map,
) -> List[Dict]:
    """Normalize and filter relationships, resolving dangling IDs via name fallback."""

    normalized_relationships = []
    rel_filter_reasons = {"invalid_from": 0, "invalid_to": 0, "invalid_both": 0, "name_matched": 0}
    for rel in result.relationships:
        normalized_rel = normalize_relationship(rel)
        from_id = normalized_rel.get("from", "")
        to_id = normalized_rel.get("to", "")
                
        # Get entity names for fallback matching
        from_name = normalized_rel.get("from_name", "") or normalized_rel.get("source_name", "")
        to_name = normalized_rel.get("to_name", "") or normalized_rel.get("target_name", "")
                
        old_from_id = from_id
        old_to_id = to_id
                
        # Use resolve_entity_id with name-based fallback
        from_id = resolve_entity_id(from_id, from_name, valid_node_ids, id_mapping, name_to_id_map, deduplicated_instances)
        to_id = resolve_entity_id(to_id, to_name, valid_node_ids, id_mapping, name_to_id_map, deduplicated_instances)
                
        # Track if we used name matching
        if (from_id != old_from_id or to_id != old_to_id) and (from_name or to_name):
            rel_filter_reasons["name_matched"] += 1
                
        normalized_rel["from"] = from_id
        normalized_rel["to"] = to_id
                
        # [FIX] 过滤自引用关系（from/to相同）
        if from_id == to_id:
            rel_filter_reasons.setdefault("self_reference", 0)
            rel_filter_reasons["self_reference"] += 1
            logger.warning(f"[RELATIONSHIP FILTER] Self-reference removed: {from_id} -> {to_id}")
            continue
                
        if from_id in entity_map:
            normalized_rel["from_name"] = entity_map[from_id].get("name", "") or entity_map[from_id].get("meta_name", "")
        if to_id in entity_map:
            normalized_rel["to_name"] = entity_map[to_id].get("name", "") or entity_map[to_id].get("meta_name", "")
                
        from_valid = from_id in valid_node_ids
        to_valid = to_id in valid_node_ids
                
        if from_valid and to_valid:
            source_layer = entity_layer_map.get(from_id, "")
            target_layer = entity_layer_map.get(to_id, "")
                    
            is_cross_layer_valid = True
            if source_layer and target_layer and source_layer != target_layer:
                try:
                    from ..relation_governance import get_relation_governance_engine
                    governance_engine = get_relation_governance_engine()
                    is_cross_layer_valid = governance_engine.is_layer_pair_allowed(source_layer, target_layer)
                except Exception as e:
                    logger.warning(f"[CROSS-LAYER VALIDATION] Failed to check layer pair {source_layer}->{target_layer}: {e}")
                    
            if is_cross_layer_valid:
                normalized_relationships.append(normalized_rel)
            else:
                rel_filter_reasons.setdefault("cross_layer_violation", 0)
                rel_filter_reasons["cross_layer_violation"] += 1
                logger.warning(f"[RELATIONSHIP FILTER] Cross-layer violation: {source_layer}->{target_layer} not allowed")
        else:
            if not from_valid and not to_valid:
                rel_filter_reasons["invalid_both"] += 1
            elif not from_valid:
                rel_filter_reasons["invalid_from"] += 1
            else:
                rel_filter_reasons["invalid_to"] += 1
            logger.warning(f"[RELATIONSHIP FILTER] Removed: {from_id}(valid={from_valid}) -> {to_id}(valid={to_valid})")

    if sum(v for k, v in rel_filter_reasons.items() if k != "name_matched") > 0:
        logger.info(f"[RELATIONSHIP FILTER] Summary: {rel_filter_reasons}, valid_node_ids count={len(valid_node_ids)}")

    return normalized_relationships




def decouple_l1_to_l2_relationships(
    normalized_relationships, deduplicated_nodes,
    entity_layer_map, valid_node_ids, entity_map,
) -> None:
    """Convert excess L1->L2 relationships to L2->L2 via proxy/Policy entities."""

    # === L1->L2 RELATIONSHIP DECOUPLING v2 ===
    # If L1->L2 relationships exceed 25% of total, convert excess to L2->L2
    # Strategy: 
    # 1. Find L2 entities managed by the L1 (via parent_id) as proxy
    # 2. If no proxy exists, create an intermediate Policy entity at L2
    # 3. Reroute L1->L2 relationships through the proxy/Policy
    l1_to_l2_rels = [r for r in normalized_relationships 
                   if entity_layer_map.get(r.get("from", ""), "") == "L1" 
                   and entity_layer_map.get(r.get("to", ""), "") == "L2"]
    total_rels = len(normalized_relationships)
    if total_rels > 0 and len(l1_to_l2_rels) / total_rels > 0.25:
        # Calculate how many to convert
        max_allowed = int(total_rels * 0.25)
        excess_count = len(l1_to_l2_rels) - max_allowed
                
        # Sort by confidence (keep higher confidence L1->L2 rels)
        l1_to_l2_rels.sort(key=lambda r: float(r.get("confidence", 0.5)), reverse=True)
                
        # Build L1->L2 mapping via parent_id
        l1_to_l2_map = {}  # L1_id -> list of L2_ids that reference it
        l2_nodes = [n for n in deduplicated_nodes if n.get("layer") == "L2"]
        l1_nodes = [n for n in deduplicated_nodes if n.get("layer") == "L1"]
        l1_name_map = {n["meta_id"]: n.get("name", "") for n in l1_nodes}
                
        for node in l2_nodes:
            pid = node.get("parent_id", "")
            if pid and entity_layer_map.get(pid, "") == "L1":
                if pid not in l1_to_l2_map:
                    l1_to_l2_map[pid] = []
                l1_to_l2_map[pid].append(node["meta_id"])
                
        converted = 0
        policy_counter = 1
                
        for rel in l1_to_l2_rels[max_allowed:]:  # Convert the lowest confidence ones
            if converted >= excess_count:
                break
            from_id = rel.get("from", "")
            to_id = rel.get("to", "")
                    
            # Strategy 1: Use existing L2 child as proxy
            replacement_l2 = ""
            if from_id in l1_to_l2_map and l1_to_l2_map[from_id]:
                for l2_id in l1_to_l2_map[from_id]:
                    if l2_id != to_id:
                        replacement_l2 = l2_id
                        break
                    
            # Strategy 2: Create intermediate Policy entity if no proxy exists
            if not replacement_l2:
                l1_name = l1_name_map.get(from_id, "")
                policy_name = f"{l1_name}_Policy"
                new_policy_id = f"L2-POLICY{policy_counter:03d}"
                policy_counter += 1
                        
                # Check if policy already exists
                existing_policy = None
                for n in l2_nodes:
                    if n.get("name", "") == policy_name:
                        existing_policy = n["meta_id"]
                        break
                        
                if existing_policy:
                    replacement_l2 = existing_policy
                else:
                    # Create new policy entity
                    policy_node = {
                        "meta_id": new_policy_id,
                        "name": policy_name,
                        "meta_class": "BusinessRule",
                        "parent_class": "BusinessRule",
                        "parent_id": from_id,
                        "layer": "L2",
                        "domain": "",
                        "description": f"Policy derived from {l1_name}",
                        "status": "active",
                        "evidence": "",
                        "attributes": []
                    }
                    deduplicated_nodes.append(policy_node)
                    valid_node_ids.add(new_policy_id)
                    entity_map[new_policy_id] = policy_node
                    entity_layer_map[new_policy_id] = "L2"
                    l2_nodes.append(policy_node)
                    replacement_l2 = new_policy_id
                    print(f"[REL DECOUPLE] Created Policy: {policy_name} ({new_policy_id})", flush=True)
                    logger.info(f"[REL DECOUPLE] Created Policy: {policy_name} ({new_policy_id})")
                    
            if replacement_l2:
                rel["from"] = replacement_l2
                if replacement_l2 in entity_map:
                    rel["from_name"] = entity_map[replacement_l2].get("name", "")
                converted += 1
                logger.info(f"[REL DECOUPLE] Rerouted: {from_id} -> {replacement_l2} -> {to_id}")
                
        if converted > 0:
            print(f"[REL DECOUPLE] Converted {converted} L1->L2 rels to L2->L2 (target: <=25%)", flush=True)
            logger.info(f"[REL DECOUPLE] Converted {converted} L1->L2 relationships to L2->L2")

    # ============================================================




def apply_hard_fixes(
    deduplicated_nodes, deduplicated_instances, normalized_relationships,
    entity_map, entity_layer_map, valid_node_ids, id_mapping,
) -> None:
    """Apply ID collision / ghost node / type correction / ECN / time-dimension fixes."""
    # === HARD FIX: ID Collision, Ghost Nodes, Type Errors, ECN ===
    # ============================================================
    # --- FIX 1: ID COLLISION - Enforce single ID per entity ---
    id_counts = Counter(n.get("meta_id", "") for n in deduplicated_nodes)
    collision_ids = {iid for iid, cnt in id_counts.items() if cnt > 1}
    if collision_ids:
        print(f"[ID COLLISION] Detected {len(collision_ids)} collision IDs: {collision_ids}", flush=True)
        # Reassign unique IDs for collided entities
        layer_counters = {}  # layer -> next counter
        for node in deduplicated_nodes:
            iid = node.get("meta_id", "")
            if iid in collision_ids:
                layer = node.get("layer", "L2")
                if layer not in layer_counters:
                    # Find max existing counter for this layer
                    existing = [n for n in deduplicated_nodes if n.get("layer") == layer]
                    max_num = 0
                    for n in existing:
                        try:
                            num = int(n.get("meta_id", "").split("-")[-1])
                            max_num = max(max_num, num)
                        except (ValueError, IndexError):
                            pass
                    layer_counters[layer] = max_num + 1
                new_id = f"{layer}-{layer_counters[layer]:03d}"
                layer_counters[layer] += 1
                old_id = iid
                node["meta_id"] = new_id
                # Update entity_map and entity_layer_map
                if old_id in entity_map:
                    entity_map[new_id] = entity_map.pop(old_id)
                entity_map[new_id] = node
                entity_layer_map[new_id] = layer
                if old_id in entity_layer_map:
                    del entity_layer_map[old_id]
                # Update valid_node_ids
                valid_node_ids.discard(old_id)
                valid_node_ids.add(new_id)
                # Update id_mapping for relationships
                id_mapping[old_id] = new_id
                print(f"[ID COLLISION] Reassigned '{node.get('name','')}' from {old_id} to {new_id}", flush=True)
                logger.info(f"[ID COLLISION] Reassigned '{node.get('name','')}' from {old_id} to {new_id}")
        # Update relationships with new IDs using name-based resolution
        # Build name -> ID map for all nodes (critical for collision resolution)
        name_to_id_map = {}
        id_to_node_map = {}
        for n in deduplicated_nodes:
            iid = n.get("meta_id", "")
            nname = n.get("name", "").strip()
            id_to_node_map[iid] = n
            if nname:
                name_to_id_map[nname.lower().strip()] = iid
        for rel in normalized_relationships:
            from_id = rel.get("from", rel.get("from_id", ""))
            to_id = rel.get("to", rel.get("to_id", ""))
            from_name = rel.get("from_name", "")
            to_name = rel.get("to_name", "")
            # Resolve from_id: check if ID matches name, if not use name-based lookup
            if from_id in id_mapping:
                # This was a collision ID - use name to find correct new ID
                if from_name:
                    name_key = from_name.lower().strip()
                    if name_key in name_to_id_map:
                        rel["from"] = name_to_id_map[name_key]
                    else:
                        rel["from"] = id_mapping[from_id]
                else:
                    rel["from"] = id_mapping[from_id]
            elif from_id in id_to_node_map and from_name:
                # Check if ID-name mismatch (ID points to different entity)
                actual_name = id_to_node_map[from_id].get("name", "").strip()
                if actual_name and actual_name.lower() != from_name.lower():
                    name_key = from_name.lower().strip()
                    if name_key in name_to_id_map:
                        rel["from"] = name_to_id_map[name_key]
            # Resolve to_id: same logic
            if to_id in id_mapping:
                if to_name:
                    name_key = to_name.lower().strip()
                    if name_key in name_to_id_map:
                        rel["to"] = name_to_id_map[name_key]
                    else:
                        rel["to"] = id_mapping[to_id]
                else:
                    rel["to"] = id_mapping[to_id]
            elif to_id in id_to_node_map and to_name:
                actual_name = id_to_node_map[to_id].get("name", "").strip()
                if actual_name and actual_name.lower() != to_name.lower():
                    name_key = to_name.lower().strip()
                    if name_key in name_to_id_map:
                        rel["to"] = name_to_id_map[name_key]
    # --- FIX 2: GHOST NODES - Create missing entities referenced in relationships ---
    all_node_ids = {n.get("meta_id", "") for n in deduplicated_nodes}
    ghost_node_ids = set()
    for rel in normalized_relationships:
        from_id = rel.get("from", rel.get("from_id", ""))
        to_id = rel.get("to", rel.get("to_id", ""))
        if from_id and from_id not in all_node_ids:
            ghost_node_ids.add((from_id, rel.get("from_name", ""), "source"))
        if to_id and to_id not in all_node_ids:
            ghost_node_ids.add((to_id, rel.get("to_name", ""), "target"))
    if ghost_node_ids:
        print(f"[GHOST NODES] Detected {len(ghost_node_ids)} ghost nodes, creating placeholders", flush=True)
        l3_counter = 100  # Start from 100 to avoid collision with existing L3 IDs
        for ghost_id, ghost_name, role in ghost_node_ids:
            # Determine layer from ID prefix
            prefix = ghost_id.split("-")[0] if "-" in ghost_id else "L3"
            if prefix == "L3":
                new_id = ghost_id if ghost_id not in all_node_ids else f"L3-GHOST{l3_counter:03d}"
                l3_counter += 1
            else:
                new_id = ghost_id
            ghost_node = {
                "meta_id": new_id,
                "name": ghost_name or f"GhostEntity_{new_id}",
                "meta_class": "Unknown",
                "parent_class": "",
                "layer": "L3",
                "domain": "",
                "description": f"Auto-created ghost node (was referenced in relationship but missing from entity list)",
                "status": "ghost",
                "evidence": "",
                "attributes": []
            }
            deduplicated_nodes.append(ghost_node)
            all_node_ids.add(new_id)
            entity_map[new_id] = ghost_node
            entity_layer_map[new_id] = "L3"
            # Update relationship to use new_id
            for rel in normalized_relationships:
                if rel.get("from", rel.get("from_id", "")) == ghost_id:
                    rel["from"] = new_id
                if rel.get("to", rel.get("to_id", "")) == ghost_id:
                    rel["to"] = new_id
            print(f"[GHOST NODES] Created: {ghost_name} ({new_id})", flush=True)
            logger.info(f"[GHOST NODES] Created placeholder: {ghost_name} ({new_id})")
    # --- FIX 3: TYPE CORRECTION ---
    for node in deduplicated_nodes:
        name = node.get("name", "")
        name_lower = name.lower().strip()
        meta_class = node.get("meta_class", "")
        layer = node.get("layer", "")
        # 3a: Material Cost / Cost concepts should NOT be Asset
        if meta_class == "Asset" and any(kw in name_lower for kw in ["cost", "price", "budget", "spend", "expense"]):
            node["meta_class"] = "Data"
            node["parent_class"] = "Data"
            print(f"[TYPE FIX] '{name}': Asset -> Data (cost/price is not physical asset)", flush=True)
            logger.info(f"[TYPE FIX] '{name}': Asset -> Data")
        # 3b: L3 entities should NOT be Capability (abstract) - change to Process or Data
        if layer == "L3" and meta_class == "Capability":
            # Determine if it's more like a process or data
            if any(kw in name_lower for kw in ["countdown", "process", "action", "flow", "handling"]):
                node["meta_class"] = "Process"
                node["parent_class"] = "Process"
            else:
                node["meta_class"] = "Data"
                node["parent_class"] = "Data"
            print(f"[TYPE FIX] '{name}' in L3: Capability -> {node['meta_class']} (L3 is physical execution layer)", flush=True)
            logger.info(f"[TYPE FIX] '{name}' in L3: Capability -> {node['meta_class']}")
        # 3c: Physical tables (Table TCP785, etc.) should be in L3, not L2
        if layer == "L2" and (name_lower.startswith("table ") or name_lower.startswith("tcp") or "table" in name_lower):
            node["layer"] = "L3"
            entity_layer_map[node["meta_id"]] = "L3"
            # Try to find a parent L2 system
            for n in deduplicated_nodes:
                if n.get("layer") == "L2" and n.get("meta_class") == "System":
                    node["parent_id"] = n["meta_id"]
                    break
            print(f"[TYPE FIX] '{name}': L2 -> L3 (physical table belongs in L3)", flush=True)
            logger.info(f"[TYPE FIX] '{name}': L2 -> L3 (physical table)")
        # 3d: Physical parts (E-valve, Seat Storage Box, etc.) must have parent_id
        if layer == "L3" and meta_class == "Asset" and not node.get("parent_id"):
            # Find best matching L2 system by name similarity
            best_parent = ""
            for n in deduplicated_nodes:
                if n.get("layer") == "L2":
                    n_name = n.get("name", "").lower()
                    # Match by keyword overlap
                    if any(kw in n_name for kw in ["noise", "cancelling", "seat", "storage", "wheel", "valve", "exhaust"]):
                        if any(kw in name_lower for kw in ["valve", "noise", "cancelling"]):
                            if "noise" in n_name or "cancelling" in n_name:
                                best_parent = n["meta_id"]
                                break
                        elif any(kw in name_lower for kw in ["seat", "storage", "box"]):
                            if "seat" in n_name or "storage" in n_name:
                                best_parent = n["meta_id"]
                                break
            # Fallback: assign to first L2 System
            if not best_parent:
                for n in deduplicated_nodes:
                    if n.get("layer") == "L2" and n.get("meta_class") == "System":
                        best_parent = n["meta_id"]
                        break
            if best_parent:
                node["parent_id"] = best_parent
                print(f"[TYPE FIX] '{name}': assigned parent_id={best_parent} (physical part must have parent)", flush=True)
                logger.info(f"[TYPE FIX] '{name}': assigned parent_id={best_parent}")
    # --- FIX 4: ECN CORE BUSINESS BACKBONE INJECTION ---
    # Check for manufacturing/supply-chain context and inject ECN entities if missing
    existing_names_lower = {n.get("name", "").lower().strip() for n in deduplicated_nodes}
    ecn_entities_to_inject = []
    ecn_relationships_to_inject = []
    # Find KDP and SIPlus for relationship anchoring
    kdp_id = ""
    siplus_id = ""
    for n in deduplicated_nodes:
        if n.get("name", "").lower() in ("kdp", "kdp system"):
            kdp_id = n["meta_id"]
        if n.get("name", "").lower() in ("siplus", "siplus system"):
            siplus_id = n["meta_id"]
    # Only inject if we have manufacturing/supply chain context
    should_inject_ecn = any(
        kw in str(existing_names_lower)
        for kw in ["kdp", "siplus", "pcr", "bom", "part", "engineering", "manufacturing"]
    )
    if should_inject_ecn:
        # Find max L2 counter
        l2_max = 0
        l3_max = 0
        for n in deduplicated_nodes:
            try:
                num = int(n.get("meta_id", "X-0").split("-")[-1])
                if n.get("layer") == "L2":
                    l2_max = max(l2_max, num)
                elif n.get("layer") == "L3":
                    l3_max = max(l3_max, num)
            except (ValueError, IndexError):
                pass
        ecn_entities = [
            ("EBOM_Structure", "Data", "L2", "Engineering Bill of Materials structure", {}),
            ("MBOM_Structure", "Data", "L2", "Manufacturing Bill of Materials structure", {}),
            ("CNote_Entity", "Data", "L2", "Change Note entity carrying delta data for ECN", {}),
            ("BOP_Process", "Process", "L2", "Bill of Process - factory workstation/sequence skeleton", {}),
            ("POS_PhysicalPosition", "Asset", "L2", "Point of Service - physical installation position", {"position": "physical_installation"}),
            ("PII_Knowledge", "Data", "L2", "Process Instruction Information - worker operation guidance", {}),
        ]
        injected_count = 0
        for ename, eclass, elayer, edesc, extra_fields in ecn_entities:
            if ename.lower() not in existing_names_lower:
                if elayer == "L2":
                    l2_max += 1
                    new_id = f"L2-ECN{l2_max:03d}"
                else:
                    l3_max += 1
                    new_id = f"L3-ECN{l3_max:03d}"
                ecn_node = {
                    "meta_id": new_id,
                    "name": ename,
                    "meta_class": eclass,
                    "parent_class": eclass,
                    "layer": elayer,
                    "domain": "MANUFACTURING",
                    "description": edesc,
                    "status": "active",
                    "evidence": "ECN backbone injection",
                    "origin_kind": "injected",
                    "provenance": {"origin": "ecn_backbone", "stage": "extract"},
                    "attributes": []
                }
                ecn_node.update(extra_fields)
                ecn_entities_to_inject.append(ecn_node)
                entity_map[new_id] = ecn_node
                entity_layer_map[new_id] = elayer
                all_node_ids.add(new_id)
                injected_count += 1
        # Build ECN relationships (corrected topology per ECN flow)
        if ecn_entities_to_inject:
            ecn_name_map = {n["name"]: n["meta_id"] for n in ecn_entities_to_inject}
            ebom_id = ecn_name_map.get("EBOM_Structure", "")
            cnote_id = ecn_name_map.get("CNote_Entity", "")
            mbom_id = ecn_name_map.get("MBOM_Structure", "")
            bop_id = ecn_name_map.get("BOP_Process", "")
            pii_id = ecn_name_map.get("PII_Knowledge", "")
            # Find production preparation system (EPop/CMP) for C-Note SENT_TO
            prod_prep_id = ""
            for n in deduplicated_nodes:
                n_name = n.get("name", "").lower()
                if n.get("layer") == "L2" and any(kw in n_name for kw in ["pop", "common", "cmp", "order handling", "production"]):
                    prod_prep_id = n["meta_id"]
                    break
            # Chain: KDP --GENERATES--> EBOM --TRACKS_DELTA--> CNote --TRANSFORMS_TO--> MBOM --CONSUMES_AT--> BOP --REQUIRES--> PII
            _inj = {
                "origin_kind": "injected",
                "provenance": {"origin": "ecn_backbone", "stage": "extract"},
            }
            if kdp_id and ebom_id:
                ecn_relationships_to_inject.append({
                    "from": kdp_id, "from_name": "KDP",
                    "relation_type": "GENERATES", "type": "GENERATES",
                    "to": ebom_id, "to_name": "EBOM_Structure",
                    "evidence_ref": "ECN: KDP generates EBOM structure",
                    **_inj,
                })
            if ebom_id and cnote_id:
                ecn_relationships_to_inject.append({
                    "from": cnote_id, "from_name": "CNote_Entity",
                    "relation_type": "TRACKS_DELTA", "type": "TRACKS_DELTA",
                    "to": ebom_id, "to_name": "EBOM_Structure",
                    "evidence_ref": "ECN: C-Note tracks delta from EBOM",
                    **_inj,
                })
            if cnote_id and prod_prep_id:
                ecn_relationships_to_inject.append({
                    "from": cnote_id, "from_name": "CNote_Entity",
                    "relation_type": "SENT_TO", "type": "SENT_TO",
                    "to": prod_prep_id, "to_name": entity_map.get(prod_prep_id, {}).get("name", "ProductionSystem"),
                    "evidence_ref": "ECN: C-Note sent to production preparation system",
                    **_inj,
                })
            if cnote_id and mbom_id:
                ecn_relationships_to_inject.append({
                    "from": cnote_id, "from_name": "CNote_Entity",
                    "relation_type": "TRANSFORMS_TO", "type": "TRANSFORMS_TO",
                    "to": mbom_id, "to_name": "MBOM_Structure",
                    "evidence_ref": "ECN: C-Note transforms to MBOM",
                    **_inj,
                })
            if mbom_id and bop_id:
                ecn_relationships_to_inject.append({
                    "from": mbom_id, "from_name": "MBOM_Structure",
                    "relation_type": "CONSUMES_AT", "type": "CONSUMES_AT",
                    "to": bop_id, "to_name": "BOP_Process",
                    "evidence_ref": "ECN: MBOM consumes at BOP workstation",
                    **_inj,
                })
            if bop_id and pii_id:
                ecn_relationships_to_inject.append({
                    "from": bop_id, "from_name": "BOP_Process",
                    "relation_type": "REQUIRES", "type": "REQUIRES",
                    "to": pii_id, "to_name": "PII_Knowledge",
                    "evidence_ref": "ECN: BOP requires PII worker guidance",
                    **_inj,
                })
            deduplicated_nodes.extend(ecn_entities_to_inject)
            normalized_relationships.extend(ecn_relationships_to_inject)
            print(f"[ECN BACKBONE] Injected {injected_count} ECN entities and {len(ecn_relationships_to_inject)} relationships", flush=True)
            logger.info(f"[ECN BACKBONE] Injected {injected_count} entities and {len(ecn_relationships_to_inject)} relationships")
    # --- FIX 5: TIME DIMENSION GOVERNANCE for L2.5 rules ---
    # Add Breakpoint_Date and Inventory_Freeze_Flag to volume/calculation rules
    for node in deduplicated_nodes:
        if node.get("layer") == "L2.5":
            name_lower = node.get("name", "").lower()
            if any(kw in name_lower for kw in ["volume", "calculation", "validation", "run-out", "runout", "effectivity", "cutoff"]):
                attrs = node.get("attributes", [])
                existing_attr_names = {a.get("name", "") if isinstance(a, dict) else str(a) for a in attrs}
                if "Breakpoint_Date" not in existing_attr_names:
                    attrs.append({"name": "Breakpoint_Date", "value": "TBD (derived from ECN effectivity date)"})
                if "Inventory_Freeze_Flag" not in existing_attr_names:
                    attrs.append({"name": "Inventory_Freeze_Flag", "value": "False (set True when lead-time calculation completes)"})
                node["attributes"] = attrs
                print(f"[TIME DIM] Added Breakpoint_Date & Inventory_Freeze_Flag to '{node.get('name','')}'", flush=True)
                logger.info(f"[TIME DIM] Added time dimension attributes to '{node.get('name','')}'")

    # --- AUTO-SUPPLEMENT POSITION/VARIANT METADATA ---
    # For nodes with position/variant in name but missing corresponding fields
    for node in deduplicated_nodes:
        if not isinstance(node, dict):
            continue
        name_lower = (node.get("name", "") or "").lower()
        if "position" in name_lower:
            if not (node.get("position") or node.get("position_code") or node.get("position_label")):
                node["position"] = name_lower.replace("position", "").strip() or "undefined"
                print(f"[AUTO-SUPPLEMENT] Added position field to '{node.get('name','')}'", flush=True)
        if "variant" in name_lower:
            if not (node.get("variant") or node.get("variant_code") or node.get("variant_name") or node.get("variant_designation")):
                node["variant"] = name_lower.replace("variant", "").replace("designation", "").strip() or "undefined"
                print(f"[AUTO-SUPPLEMENT] Added variant field to '{node.get('name','')}'", flush=True)

    # --- L3 PARENT_ID -> RELATIONSHIP DERIVATION ---
    # Auto-generate explicit relationships from L3 instances' parent_id
    # This ensures L3 instances always have at least one relationship
    # connecting them to their parent L2/L2.5 node
    l3_parent_rels_added = 0
    existing_rel_pairs = set()
    for r in normalized_relationships:
        rfrom = r.get("from", r.get("from_id", ""))
        rto = r.get("to", r.get("to_id", ""))
        existing_rel_pairs.add((rfrom, rto))
    for inst in deduplicated_instances:
        inst_id = inst.get("meta_id", "")
        parent_id = inst.get("parent_id", "")
        if not inst_id or not parent_id:
            continue
        if parent_id not in valid_node_ids:
            continue
        if (parent_id, inst_id) not in existing_rel_pairs:
            parent_node = entity_map.get(parent_id, {})
            parent_layer = parent_node.get("layer", "")
            rel = {
                "from": parent_id,
                "to": inst_id,
                "relation_type": "HAS_COMPONENT",
                "from_name": parent_node.get("name", ""),
                "to_name": inst.get("name", ""),
                "evidence": inst.get("evidence", {}),
                "source_file": inst.get("source_file", ""),
            }
            normalized_relationships.append(rel)
            existing_rel_pairs.add((parent_id, inst_id))
            l3_parent_rels_added += 1
    if l3_parent_rels_added > 0:
        print(f"[L3 PARENT REL] Added {l3_parent_rels_added} parent->instance relationships from parent_id", flush=True)
        logger.info(f"[L3 PARENT REL] Added {l3_parent_rels_added} parent->instance relationships from parent_id")


# --- Entity enhance / L0 validation / dedup / index builders -----------------


def enhance_entity(entity: Dict, layer: str, source_file_str: str) -> Dict:
    """Tag layer + source_file, normalize type and evidence.

    Extracted verbatim from the inline closure formerly defined inside
    ``extract_async_content`` (kept as a module-level pure function so the
    orchestrator no longer carries a closure over ``source_file_str``).
    """
    entity["layer"] = layer
    entity["source_file"] = source_file_str

    # P1-03: Normalize entity type to fix duplicate classification
    if layer in ("L1", "L2", "L2.5"):
        entity = normalize_entity_type(entity, layer)

    if "evidence" in entity and entity["evidence"]:
        entity["evidence"] = normalize_evidence(entity["evidence"])
        ev = entity["evidence"]
        if isinstance(ev, dict):
            context = ev.get("context", "")
            raw_text = ev.get("raw_text", "")
            if context and "description" in entity and len(entity["description"]) < 200:
                entity["description"] = f"{entity['description']}. Context: {context[:200]}"
            elif context and "description" not in entity:
                entity["description"] = context[:300]

    if layer == "L1" and "parent_class" not in entity:
        entity["parent_class"] = entity.get("meta_class", "System")

    return entity


def validate_l0_entities(
    l0_entities: List[Dict], valid_l0_classes: set, source_file_str: str,
) -> List[Dict]:
    """Validate L0 entities: exact class names stay L0, domain-specific
    concepts and invalid names are demoted to L2 (verbatim from the inline
    L0 validation loop)."""
    valid_l0_classes_lower = {c.lower() for c in valid_l0_classes}
    all_entities: List[Dict] = []

    for entity in l0_entities:
        entity_name = entity.get("name", "").strip()
        entity_meta_class = entity.get("meta_class", "").strip()

        if entity_name.lower() in valid_l0_classes_lower:
            for valid_class in valid_l0_classes:
                if valid_class.lower() == entity_name.lower():
                    entity["name"] = valid_class
                    break
            all_entities.append(enhance_entity(entity, "L0", source_file_str))
        elif entity_meta_class.lower() in valid_l0_classes_lower:
            # Domain-specific concepts with L0 meta_class are CONCRETE
            # implementations, not abstract subtypes -> L2, NOT L1
            logger.warning(f"[L0 VALIDATION] Entity name '{entity_name}' is a domain-specific concept (not abstract L0). Moving to L2 with parent_class='{entity_meta_class}'.")
            entity["layer"] = "L2"
            entity["meta_id"] = entity["meta_id"].replace("L0-", "L2-") if "meta_id" in entity else ""
            entity["parent_class"] = entity_meta_class
            all_entities.append(enhance_entity(entity, "L2", source_file_str))
        else:
            logger.warning(f"[L0 VALIDATION] Entity '{entity_name}' is not a valid L0 class, moved to L2")
            entity["layer"] = "L2"
            entity["meta_id"] = entity["meta_id"].replace("L0-", "L2-") if "meta_id" in entity else ""
            all_entities.append(enhance_entity(entity, "L2", source_file_str))

    return all_entities


def deduplicate_nodes(
    all_entities: List[Dict], counter: dict, id_mapping: dict, seen_ids: dict,
) -> List[Dict]:
    """Deduplicate all layer entities into ``deduplicated_nodes``, assigning
    prefixed sequential IDs and maintaining ``id_mapping`` (verbatim from the
    inline dedup loop)."""
    deduplicated_nodes: List[Dict] = []

    for entity in all_entities:
        layer = entity.get("layer", "L1")
        name = entity.get("name", "")
        meta_class = entity.get("meta_class", "")
        original_id = entity.get("meta_id", "")
        evidence = entity.get("evidence")
        if isinstance(evidence, dict):
            source_file = entity.get("source_file", "") or evidence.get("source_file", "")
        else:
            source_file = entity.get("source_file", "")

        # P1-02: Filter noise entities (skip L0 auto-supplemented)
        if layer != "L0" and is_noise_entity(entity):
            logger.warning(f"[NOISE FILTER] Filtered noise entity: name='{name}', layer={layer}")
            if original_id:
                id_mapping[original_id] = ""
            continue

        if layer in ("L0", "L1"):
            entity_key = (layer, name, meta_class)
        else:
            entity_key = (layer, name, meta_class, source_file) if source_file else (layer, name, meta_class)

        if entity_key in seen_ids:
            existing = seen_ids[entity_key]
            if original_id:
                id_mapping[original_id] = existing["meta_id"]
            if "attributes" in entity and entity["attributes"]:
                if "attributes" not in existing or not existing["attributes"]:
                    existing["attributes"] = entity["attributes"]
                else:
                    existing_attr_names = {a.get("name", "") for a in existing["attributes"]}
                    for attr in entity["attributes"]:
                        if attr.get("name", "") not in existing_attr_names:
                            existing["attributes"].append(attr)
            if "description" in entity and entity["description"]:
                if "description" not in existing or not existing["description"]:
                    existing["description"] = entity["description"]
            continue

        prefix_map = {"L0": "L0-", "L1": "L1-", "L2": "L2-", "L2.5": "L2.5-", "L3": "L3-"}
        prefix = prefix_map.get(layer, "L1-")
        new_id = f"{prefix}{counter[layer]:03d}"
        counter[layer] += 1

        entity["meta_id"] = new_id
        if original_id:
            id_mapping[original_id] = new_id

        seen_ids[entity_key] = entity
        deduplicated_nodes.append(entity)

    return deduplicated_nodes


def deduplicate_instances(
    instances: List[Dict], counter: dict, id_mapping: dict, seen_ids: dict,
) -> List[Dict]:
    """Deduplicate L3 instances, assigning sequential L3- IDs.

    Mirrors the structure of :func:`deduplicate_nodes` for L3: each unique
    (non-duplicate) instance is appended to the result inside the loop,
    so empty instance lists return ``[]`` without referencing an unbound
    ``entity`` variable.
    """
    deduplicated_instances: List[Dict] = []
    seen_instance_ids: dict = {}

    for entity in instances:
        inst_source_file = entity.get("source_file", "") or entity.get("evidence", {}).get("source_file", "") if isinstance(entity.get("evidence"), dict) else ""
        entity_key = ("L3", entity.get("name", ""), entity.get("meta_class", ""), inst_source_file) if inst_source_file else ("L3", entity.get("name", ""), entity.get("meta_class", ""))
        original_id = entity.get("meta_id", "")

        if entity_key in seen_instance_ids:
            existing = seen_instance_ids[entity_key]
            if original_id:
                id_mapping[original_id] = existing["meta_id"]
            if "attributes" in entity and entity["attributes"]:
                if "attributes" not in existing or not existing["attributes"]:
                    existing["attributes"] = entity["attributes"]
                else:
                    existing_attr_names = {a.get("name", "") for a in existing["attributes"]}
                    for attr in entity["attributes"]:
                        if attr.get("name", "") not in existing_attr_names:
                            existing["attributes"].append(attr)
            continue

        new_id = f"L3-{counter['L3']:03d}"
        counter["L3"] += 1

        entity["meta_id"] = new_id
        if original_id:
            id_mapping[original_id] = new_id

        seen_instance_ids[entity_key] = entity
        deduplicated_instances.append(entity)

    return deduplicated_instances


def build_entity_maps(
    nodes: List[Dict], instances: List[Dict],
):
    """Build layer map / entity map / L0-class-to-id lookup used for
    cross-layer validation. Must run BEFORE L1->L2 reclassification."""
    entity_layer_map: Dict[str, str] = {}
    entity_map: Dict[str, Dict] = {}
    l0_class_to_id: Dict[str, str] = {}
    for node in nodes + instances:
        entity_layer_map[node["meta_id"]] = node.get("layer", "")
        entity_map[node["meta_id"]] = node
        if node.get("layer") == "L0":
            l0_class_to_id[node.get("name", "").lower()] = node["meta_id"]
    return entity_layer_map, entity_map, l0_class_to_id


def autogenerate_parent_ids(nodes: List[Dict], l0_class_to_id: Dict[str, str]) -> None:
    """Auto-generate parent_id for L1 entities referencing an L0 class."""
    for node in nodes:
        if node.get("layer") == "L1" and "parent_id" not in node:
            parent_class = node.get("parent_class", node.get("meta_class", "")).lower()
            if parent_class in l0_class_to_id:
                node["parent_id"] = l0_class_to_id[parent_class]


def autogenerate_parent_classes(nodes: List[Dict]) -> None:
    """Auto-generate parent_class for L2 entities."""
    for node in nodes:
        if node.get("layer") == "L2" and "parent_class" not in node:
            node["parent_class"] = node.get("meta_class", "BusinessScenario")


def build_name_to_id_map(nodes: List[Dict], instances: List[Dict]) -> dict:
    """Build name->id lookup for fallback matching when ID mapping fails
    (LLM sometimes emits non-standard IDs like L1-SYS001)."""
    name_to_id_map: dict = {}
    for node in nodes + instances:
        node_name = (node.get("name") or node.get("meta_name") or "").strip().lower()
        if node_name:
            layer = node.get("layer", "")
            name_to_id_map[(layer, node_name)] = node["meta_id"]
            if node_name not in name_to_id_map:
                name_to_id_map[node_name] = node["meta_id"]
    return name_to_id_map


def apply_parent_id_fixes(
    nodes: List[Dict], instances: List[Dict], valid_node_ids: set,
    id_mapping: dict, name_to_id_map: dict, deduplicated_instances: List[Dict],
) -> None:
    """Resolve dangling parent_id references via ID mapping, then name-based
    fallback; drop the parent_id if it still cannot be resolved."""
    for node in nodes:
        parent_id = node.get("parent_id", "")
        if parent_id and parent_id not in valid_node_ids:
            parent_name = node.get("parent_name", "") or node.get("parent_class", "")
            new_parent_id = resolve_entity_id(parent_id, parent_name, valid_node_ids, id_mapping, name_to_id_map, deduplicated_instances)
            if new_parent_id in valid_node_ids:
                node["parent_id"] = new_parent_id
                logger.info(f"[PARENT FIX] Updated parent_id from {parent_id} to {new_parent_id}")
            else:
                logger.warning(f"[PARENT FIX] Removing invalid parent_id: {parent_id} for node {node.get('meta_id', '')}")
                node.pop("parent_id", None)

    for instance in instances:
        parent_id = instance.get("parent_id", "")
        if parent_id and parent_id not in valid_node_ids:
            parent_name = instance.get("parent_name", "") or instance.get("parent_class", "")
            new_parent_id = resolve_entity_id(parent_id, parent_name, valid_node_ids, id_mapping, name_to_id_map, deduplicated_instances)
            if new_parent_id in valid_node_ids:
                instance["parent_id"] = new_parent_id
                logger.info(f"[PARENT FIX] Updated parent_id from {parent_id} to {new_parent_id} for instance {instance.get('meta_id', '')}")
            else:
                logger.warning(f"[PARENT FIX] Removing invalid parent_id: {parent_id} for instance {instance.get('meta_id', '')}")
                instance.pop("parent_id", None)


__all__ = [
    "reclassify_l1_to_l2",
    "reclassify_l2_to_l1",
    "apply_layer_quality_filter",
    "normalize_and_filter_relationships",
    "decouple_l1_to_l2_relationships",
    "apply_hard_fixes",
    "enhance_entity",
    "validate_l0_entities",
    "deduplicate_nodes",
    "deduplicate_instances",
    "build_entity_maps",
    "autogenerate_parent_ids",
    "autogenerate_parent_classes",
    "build_name_to_id_map",
    "apply_parent_id_fixes",
    "L1_TO_L2_KEYWORDS",
    "L1_RETENTION_PATTERNS",
]