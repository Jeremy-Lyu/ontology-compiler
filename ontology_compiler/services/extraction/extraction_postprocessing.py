"""Post-processing helpers for the unified extractor.

Pure constants and functions extracted from the inline body of
``extract_async_content`` in ``true_unified_extractor``. They normalize entity
types, filter noise entities, infer L1 subclasses, and normalize relationship
types before deduplication and layer re-classification.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# P1-02: Noise entity detection
NOISE_GENERIC_WORDS = {
    "presentation", "document", "content", "introduction", "overview",
    "slide", "title", "header", "footer", "note", "comment",
}

NOISE_TECH_ATTRS = {
    "dateformat", "securityclass", "phasetransition", "timestamp",
    "version", "status", "id", "uid", "guid",
}


def is_noise_entity(entity: dict) -> bool:
    """Check if entity is noise (dates, slide numbers, generic words)."""
    name = str(entity.get("name", "")).strip()
    if not name:
        return True

    if name.isdigit():
        return True

    if name.lower().startswith("slide "):
        return True

    if name.lower() in NOISE_GENERIC_WORDS:
        return True

    if name.lower() in NOISE_TECH_ATTRS:
        return True

    if len(name) < 2:
        return True

    if re.match(r"^\d{6,8}$", name):
        return True
    if re.match(r"^\d{4}-\d{2}-\d{2}", name):
        return True

    return False


# P1-03 + P2-02: Entity type overrides and L1 subclass inference
ENTITY_TYPE_OVERRIDES = {
    "EBOM": ("Data", "Document_Artifact"),
    "MBOM": ("Data", "Document_Artifact"),
    "BOP": ("Process", "Business_Process"),
    "BOM": ("Data", "Document_Artifact"),
    "BOE": ("Data", "Document_Artifact"),
    "Teamcenter": ("System", "Enterprise_Platform"),
    "SAP": ("System", "Enterprise_Platform"),
    "MES": ("System", "Enterprise_Platform"),
    "KDP": ("System", "Enterprise_Platform"),
    "CMP": ("System", "Enterprise_Platform"),
    "PLUS": ("System", "Enterprise_Platform"),
    "PIF": ("System", "Microservice_App"),
    "MPP": ("System", "Enterprise_Platform"),
    "Delivery Unit": ("Data", "Product_Structure_Node"),
    "Engine": ("Asset", "Product_Definition"),
    "Assembly": ("Asset", "Product_Definition"),
    "Component": ("Asset", "Product_Definition"),
    "Position": ("Data", "Structural_Position"),
    "Variant": ("Data", "Variant_Definition"),
    "Cyl Head Assy": ("Asset", "Assembly_Node"),
}

L1_SUBCLASS_INFERENCE = {
    "System": {
        "Enterprise_Platform": ["teamcenter", "sap", "kdp", "cmp", "plus", "mes", "mpp", "ohf", "aro", "hercules", "pif", "tc plm", "plm", "tc plm"],
        "Microservice_App": ["oss", "sbs toolbox", "sbs", "microservice", "api", "service"],
        "Test_Environment": ["qnx", "vteststudio", "test", "simulation", "hil"],
        "External_Portal": ["ipca", "portal", "codev", "supplier", "external"],
    },
    "Data": {
        "Document_Artifact": ["bom", "mbom", "ebom", "mds", "document", "sheet", "manual", "report", "form", "boe"],
        "Rule_Logic": ["variant", "configuration", "rule", "expression", "logic"],
        "Data_Record": ["record", "data", "item", "part", "address"],
        "Table_Schema": ["schema", "table", "field", "column", "structure"],
        "Structural_Position": ["position", "pos"],
        "Variant_Definition": ["variant", "v1", "v2"],
        "Product_Structure_Node": ["delivery unit", "du", "assembly", "component"],
    },
    "Actor": {
        "Internal_Department": ["engineering", "manufacturing", "department", "division", "team"],
        "Business_Function": ["specialist", "admin", "manager", "engineer", "role", "function", "swe", "ccb"],
        "Equipment_Infra": ["robot", "workstation", "server", "infrastructure", "equipment", "desk"],
    },
    "Asset": {
        "Product_Definition": ["part", "component", "hardware", "drawing", "design", "caliper", "actuator", "brake", "cover", "engine", "assembly"],
        "Software_Asset": ["firmware", "software", "algorithm", "model", "code", "package"],
        "Tooling_Asset": ["mold", "fixture", "tooling", "die", "jig"],
        "Consumable_Material": ["material", "adhesive", "paint", "consumable", "batch"],
        "Vehicle_Instance": ["vin", "vehicle", "car", "truck"],
    },
    "Process": {
        "Business_Process": ["order", "change", "approval", "workflow", "process", "release", "planning"],
    },
    "Capability": {
        "L1_Domain": ["product development", "supply chain", "manufacturing", "aftersales"],
        "L2_SubDomain": ["product structure", "verification", "logistics", "quality"],
        "L3_Actionable": ["management", "handling", "processing", "documentation", "tracking"],
    },
}


def infer_l1_subclass(entity: dict) -> str:
    """Infer L1 subclass based on entity name keywords."""
    name = str(entity.get("name", "")).lower().strip()
    parent_class = entity.get("parent_class", entity.get("meta_class", ""))

    if parent_class in L1_SUBCLASS_INFERENCE:
        for subclass, keywords in L1_SUBCLASS_INFERENCE[parent_class].items():
            if any(kw in name for kw in keywords):
                return subclass
    return parent_class


def normalize_entity_type(entity: dict, layer: str = "") -> dict:
    """Normalize entity type to fix duplicate classification and infer L1 subclass."""
    name = str(entity.get("name", "")).strip()

    for override_name, (correct_class, correct_subclass) in ENTITY_TYPE_OVERRIDES.items():
        if name.lower() == override_name.lower():
            old_class = entity.get("meta_class", "")
            if old_class != correct_class:
                logger.info(f"[TYPE NORMALIZE] Entity '{name}': {old_class} -> {correct_class}")
            entity["meta_class"] = correct_class
            entity["parent_class"] = correct_class
            entity["meta_subclass"] = correct_subclass
            return entity

    if "meta_subclass" not in entity:
        inferred_subclass = infer_l1_subclass(entity)
        if inferred_subclass != entity.get("parent_class", ""):
            logger.info(f"[L1 SUBCLASS] Entity '{name}': inferred subclass '{inferred_subclass}'")
        entity["meta_subclass"] = inferred_subclass

    if layer == "L2.5":
        if any(k in name.lower() for k in ["variant", "position", "alignment", "occurrence", "source structure", "source_structure"]):
            entity["meta_class"] = entity.get("meta_class", "Data") or "Data"
            entity.setdefault("meta_subclass", "Rule_Logic")

    return entity


def normalize_evidence(evidence: Any) -> Dict:
    """Coerce evidence into the standard dict shape."""
    if isinstance(evidence, dict):
        return evidence
    if hasattr(evidence, "model_dump"):
        return evidence.model_dump()
    if isinstance(evidence, str):
        return {"source_text": evidence, "confidence": 0.7}
    if isinstance(evidence, (list, tuple)):
        return {"items": evidence, "confidence": 0.7}
    return {"confidence": 0.0}


# P2-03: Relation type normalization (lowercase -> UPPERCASE standard)
RELATION_TYPE_NORMALIZATION = {
    "contains": "CONTAINS",
    "has_part": "CONTAINS",
    "has": "CONTAINS",
    "includes": "CONTAINS",
    "comprises": "CONTAINS",
    "consists_of": "CONTAINS",
    "part_of": "PART_OF",
    "is_part_of": "PART_OF",
    "instance_of": "INSTANCE_OF",
    "is_instance_of": "INSTANCE_OF",
    "has_variant": "HAS_VARIANT",
    "couples": "CONTAINS",
    "mirrors": "RELATED_TO",
    "uses": "USES",
    "utilizes": "USES",
    "consumes": "USES",
    "accesses": "USES",
    "reads": "USES",
    "writes": "USES",
    "implements": "IMPLEMENTS",
    "realizes": "IMPLEMENTS",
    "produces": "GENERATES",
    "outputs": "GENERATES",
    "delivers": "GENERATES",
    "generates_output": "GENERATES",
    "generates": "GENERATES",
    "creates": "GENERATES",
    "enables": "ENABLES",
    "critical_for": "ENABLES",
    "supports": "SUPPORTS",
    "related_to": "RELATED_TO",
    "associated_with": "RELATED_TO",
    "connected_to": "RELATED_TO",
    "linked_to": "RELATED_TO",
    "aligned_with": "RELATED_TO",
    "connects": "RELATED_TO",
    "links": "RELATED_TO",
    "integrates_with": "INTEGRATES_WITH",
    "integrated_with": "INTEGRATES_WITH",
    "integrates": "INTEGRATES_WITH",
    "interacts_with": "INTERACTS_WITH",
    "communicates_with": "INTERACTS_WITH",
    "collaborates_with": "INTERACTS_WITH",
    "interacts": "INTERACTS_WITH",
    "extends": "EXTENDS",
    "validates": "VALIDATES",
    "governs_data": "GOVERNS_DATA",
    "governs": "GOVERNS_DATA",
    "defines": "GOVERNS_DATA",
    "describes": "GOVERNS_DATA",
    "maintains": "MAINTAINS",
    "controls": "MAINTAINS",
    "manages": "MANAGES",
    "managed_by": "MAINTAINS",
    "handles": "MAINTAINS",
    "steers": "MAINTAINS",
    "replaces": "REPLACES",
    "supersedes": "SUPERSEDES",
    "same_as": "SAME_AS",
    "depends_on": "DEPENDS_ON",
    "requires": "DEPENDS_ON",
    "relies_on": "DEPENDS_ON",
    "based_on": "DEPENDS_ON",
    "modifies": "MODIFIES",
    "updates": "MODIFIES",
    "results_in": "RESULTS_IN",
    "references": "REFERENCES",
    "traces_to": "TRACES_TO",
    "sends_to": "SENDS_TO",
    "sends": "SENDS_TO",
    "sends_data": "SENDS_TO",
    "transfers": "TRANSFERS_TO",
    "transfers_to": "TRANSFERS_TO",
    "transfers_data": "TRANSFERS_TO",
    "stored_in": "STORED_AT",
    "stored_at": "STORED_AT",
    "executes": "EXECUTES",
    "calls": "EXECUTES",
    "invokes": "EXECUTES",
    "deploys_to": "DEPLOYS_TO",
    "derived_from": "DERIVED_FROM",
    "owns": "OWNS",
    "operates": "OPERATES",
    "responsible_for": "RESPONSIBLE_FOR",
    "assigned_to": "ASSIGNED_TO",
    "participates_in": "PARTICIPATES_IN",
    "performs": "EXECUTES",
    "processes": "EXECUTES",
    "transforms": "MODIFIES",
    "affects": "RELATED_TO",
}

ALLOWED_RELATION_TYPES = {
    "CONTAINS", "PART_OF", "INSTANCE_OF", "OWNS", "OPERATES",
    "RESPONSIBLE_FOR", "ASSIGNED_TO", "PARTICIPATES_IN", "EXECUTES",
    "MODIFIES", "RELATED_TO", "USES", "IMPLEMENTS", "DEPENDS_ON",
    "MANAGES", "GENERATES", "VALIDATES", "GOVERNS_DATA", "SUPPORTS",
    "INTEGRATES_WITH", "INTERACTS_WITH", "EXTENDS", "ENABLES",
    "MAINTAINS", "REPLACES", "SUPERSEDES", "SAME_AS", "RESULTS_IN",
    "REFERENCES", "TRACES_TO", "SENDS_TO", "TRANSFERS_TO", "STORED_AT",
    "DERIVED_FROM",
}


def normalize_relationship(rel: Dict) -> Dict:
    """Normalize relationship field names and relation type to the canonical UPPERCASE set."""
    if isinstance(rel, dict):
        if "source_id" in rel and "target_id" in rel:
            rel["from"] = rel.pop("source_id")
            rel["to"] = rel.pop("target_id")

        rel_type = rel.get("relation_type", rel.get("type", "RELATED_TO"))
        rel_type_lower = rel_type.lower()
        if rel_type_lower == "unknown":
            normalized_type = "RELATED_TO"
        else:
            normalized_type = RELATION_TYPE_NORMALIZATION.get(rel_type_lower, rel_type.upper())

        rel["type"] = normalized_type
        rel["relation_type"] = normalized_type

        rel.setdefault("confidence", 0.7)
        rel.setdefault("evidence_ref", "")
    return rel


# --- Layer quality filter v2 ---
# Based on OEM ontology meta-model requirements:
# 1. L1 = Fixed enumeration dictionary (no specific business names)
# 2. L2 = Logical entities (macro systems, departments, processes)
# 3. L3 = Physical instances (tables, features, part instances)
# 4. Asset priority: physical parts must be Asset class

# L1 FIXED ENUMERATION: Only these abstract subtypes are allowed in L1
L1_ALLOWED_ENUMERATION = {
    "internal_department", "business_function", "external_partner",
    "master_data", "transaction_data", "document",
    "supplychain_system", "manufacturing_system", "retail_system",
    "engineering_system", "quality_system",
    "supplychain_process", "manufacturing_process", "engineering_process",
    "costmanagement", "changemanagement", "qualitymanagement",
    "requirementsmanagement", "riskmanagement", "performance management",
    "dealermanagement", "documentmanagement", "procurementmanagement",
    "stakeholdermanagement", "activitymanagement", "financialmanagement",
    "vehicle_asset", "part_asset", "tooling_asset",
}

# Generic terms that are too vague to be useful entities
GENERIC_L1_TERMS = {
    "system", "process", "capability", "actor", "asset", "data",
    "concept", "governance", "management", "strategy",
    "dealer", "benchmark process", "benchmark",
    "creator", "currency", "part", "parts",
    "businessscenario", "business rule", "mapping rule",
}

# Generic terms that should NOT be L2 entities
GENERIC_L2_TERMS = {
    "businesscase", "business case", "data access", "data management",
    "current solution", "new solution", "solution",
    "retail systems", "retail system",
    "businessscenario", "business rule", "mapping rule",
    "benchmark",
}

# L2->L3 DOWNGRADE: These L2 entities are too micro/physical for L2
L2_TO_L3_PATTERNS = [
    r".*[Tt]able$",
    r"^TCP\d+.*",
    r"^TCP785$",
    r"^[Uu]se [Uu]p.*",
    r".*[Cc]ountdown.*",
    r".*[Cc]ustomization$",
    r".*[Ee]xport$",
    r"^[Aa]ctivity [Cc]reation$",
    r"^[Dd]ata [Ee]xport$",
]

# L1 TOO SPECIFIC: Patterns for L1 entities that should be L2
L1_TOO_SPECIFIC_PATTERNS = [
    r".*\d+\.\d+.*",  # Version numbers
    r".*[Tt]emplate.*",
    r".*[Cc]hecklist.*",
    r".*[Nn]oise [Cc]ancell.*",
    r".*[Ww]heel [Ll]ocal.*",
    r"^\d+.*inch.*",
    r"^\d+-inch.*",
]

# ASSET DETECTION: Keywords that indicate physical assets
ASSET_KEYWORDS = {
    "wheel", "seat", "valve", "engine", "motor", "battery",
    "chassis", "body", "door", "mirror", "tire", "axle",
    "tooling", "mold", "fixture", "gantry", "robot",
    "vin", "vehicle", "car", "truck",
}

# L1 ORG/TEAM DOWNGRADE: L1 entities with Team/Department should go to L2
L1_ORG_KEYWORDS = {"team", "department", "committee", "board", "group"}

# Patterns for specific part/material instances -> L3
PART_INSTANCE_PATTERNS = [
    r"^[A-Z]{2}\d{2}.*",
    r".*[Ww]heel$",
    r".*[Bb]ox$",
    r".*[Vv]alve$",
    r".*[Ee][Vv]alve$",
]


def correct_meta_class(name: str, current_class: str) -> str:
    """Correct meta_class based on entity name patterns."""
    name_lower = name.lower().strip()
    for kw in ASSET_KEYWORDS:
        if kw in name_lower and current_class not in ("Asset",):
            return "Asset"
    if name_lower.endswith("system") and current_class != "System":
        return "System"
    if name_lower.endswith("process") and current_class not in ("Process",):
        return "Process"
    actor_keywords = ["engineer", "buyer", "manager", "analyst", "specialist",
                      "coordinator", "planner", "operator", "director", "officer",
                      "consultant", "assistant", "supervisor", "leader", "head"]
    if (name_lower.endswith("department") or name_lower.endswith("team") or
            name_lower.endswith("organization") or name_lower.endswith("group") or
            name_lower.endswith("division") or name_lower.endswith("unit") or
            any(kw in name_lower for kw in actor_keywords)):
        if current_class != "Actor":
            return "Actor"
    data_keywords = ["time", "window", "date", "period", "duration",
                     "value", "calculation", "fluctuation", "quantity",
                     "amount", "volume", "rate", "threshold", "limit",
                     "lead time", "cycle time", "forecast"]
    if any(kw in name_lower for kw in data_keywords):
        if current_class in ("System", "Asset", "Actor"):
            return "Data"
    if name_lower.endswith("rule") and current_class not in ("BusinessRule", "ValidationRule"):
        return "BusinessRule"
    system_keywords = ["calculator", "converter", "portal", "platform",
                       "tool", "application", "database", "registry",
                       "tracker", "dashboard", "module", "engine"]
    if any(kw in name_lower for kw in system_keywords):
        if current_class not in ("System",):
            return "System"
    return current_class


def resolve_entity_id(
    raw_id: str,
    raw_name: str,
    valid_node_ids: set,
    id_mapping: dict,
    name_to_id_map: dict,
    instances: list,
) -> str:
    """Resolve entity ID with fallback to name-based matching."""
    if raw_id in valid_node_ids:
        return raw_id
    mapped_id = id_mapping.get(raw_id, "")
    if mapped_id and mapped_id in valid_node_ids:
        return mapped_id
    if raw_id.startswith("L0-"):
        l1_id = "L1-" + raw_id[3:]
        if l1_id in valid_node_ids:
            return l1_id
        mapped_l1 = id_mapping.get(l1_id, "")
        if mapped_l1 and mapped_l1 in valid_node_ids:
            return mapped_l1
    if raw_id.startswith("SC-ENT-"):
        stripped_id = raw_id[7:]
        if stripped_id in valid_node_ids:
            return stripped_id
        mapped_stripped = id_mapping.get(stripped_id, "")
        if mapped_stripped and mapped_stripped in valid_node_ids:
            return mapped_stripped
    if raw_id.startswith("SC_L"):
        m = re.match(r"^SC_L([0-9.]+)_(\d+)$", raw_id)
        if m:
            layer = m.group(1)
            num = int(m.group(2))
            candidate_id = f"L{layer}-{num:03d}"
            if candidate_id in valid_node_ids:
                return candidate_id
            mapped_cand = id_mapping.get(candidate_id, "")
            if mapped_cand and mapped_cand in valid_node_ids:
                return mapped_cand
    if raw_id.startswith("L3-") and raw_id != "L3-001":
        m = re.match(r"^L3-[A-Z]+-(\d+)$", raw_id)
        if m:
            candidate_id = f"L3-{int(m.group(1)):03d}"
            if candidate_id in valid_node_ids:
                return candidate_id
            mapped_cand = id_mapping.get(candidate_id, "")
            if mapped_cand and mapped_cand in valid_node_ids:
                return mapped_cand
    if raw_id.startswith("L3-") and raw_name:
        name_lower = raw_name.strip().lower()
        for inst in instances:
            inst_name = (inst.get("name") or "").strip().lower()
            if inst_name and inst_name == name_lower:
                return inst["meta_id"]
    if raw_name:
        name_lower = raw_name.strip().lower()
        if name_lower in name_to_id_map:
            return name_to_id_map[name_lower]
    return raw_id


def enforce_l3_parents(nodes: list, relationships: list) -> None:
    """Ensure all L3 entities have a parent_id, inferring from relationships or closest L2."""
    relationships = relationships or []
    l3_without_parent = [n for n in nodes if n.get("layer") == "L3" and not n.get("parent_id")]
    if not l3_without_parent:
        return
    l2_nodes = [n for n in nodes if n.get("layer") == "L2"]
    l2_name_map = {}
    for n in l2_nodes:
        name_lower = n.get("name", "").lower().strip()
        if name_lower:
            l2_name_map[name_lower] = n["meta_id"]

    for l3_node in l3_without_parent:
        l3_name = l3_node.get("name", "")
        l3_meta_class = l3_node.get("meta_class", "")
        l3_id = l3_node.get("meta_id", "")

        inferred_parent = ""
        l2_ids = {n["meta_id"] for n in l2_nodes}
        for rel in relationships:
            rel_from = rel.get("from", rel.get("from_id", ""))
            rel_to = rel.get("to", rel.get("to_id", ""))
            if rel_from == l3_id and rel_to in l2_ids:
                inferred_parent = rel_to
                break
            if rel_to == l3_id and rel_from in l2_ids:
                inferred_parent = rel_from
                break

        if not inferred_parent:
            l3_name_lower = l3_name.lower()
            for l2_name, l2_id in l2_name_map.items():
                if l2_name in l3_name_lower or l3_name_lower in l2_name:
                    inferred_parent = l2_id
                    break

        if not inferred_parent:
            for n in l2_nodes:
                if n.get("meta_class", "") == l3_meta_class:
                    inferred_parent = n["meta_id"]
                    break

        if not inferred_parent and l2_nodes:
            inferred_parent = l2_nodes[0]["meta_id"]

        if inferred_parent:
            l3_node["parent_id"] = inferred_parent
            print(f"[PARENT FIX] Assigned parent_id {inferred_parent} to L3 '{l3_name}' ({l3_id})", flush=True)
            logger.info(f"[PARENT FIX] Assigned parent_id {inferred_parent} to L3 '{l3_name}'")
        else:
            logger.warning(f"[PARENT FIX] No suitable parent found for L3 '{l3_name}' ({l3_id})")


__all__ = [
    "is_noise_entity",
    "normalize_entity_type",
    "infer_l1_subclass",
    "normalize_evidence",
    "normalize_relationship",
    "correct_meta_class",
    "resolve_entity_id",
    "enforce_l3_parents",
    "ENTITY_TYPE_OVERRIDES",
    "L1_SUBCLASS_INFERENCE",
    "RELATION_TYPE_NORMALIZATION",
    "ALLOWED_RELATION_TYPES",
    "L1_ALLOWED_ENUMERATION",
    "GENERIC_L1_TERMS",
    "GENERIC_L2_TERMS",
    "L2_TO_L3_PATTERNS",
    "L1_TOO_SPECIFIC_PATTERNS",
    "ASSET_KEYWORDS",
    "L1_ORG_KEYWORDS",
    "PART_INSTANCE_PATTERNS",
]