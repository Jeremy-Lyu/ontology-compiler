"""Extraction validation machinery extracted from ``TrueUnifiedExtractor``.

The whole validation cluster (``_infer_entity_class``, ``_validate_extraction``,
``_validate_response_semantics``, ``_add_validation_feedback``,
``_add_governance_feedback``, ``_validate_with_schema_registry``) never
referenced ``self``, so it now lives here as plain module functions
(2026-08-11).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    from core.schema_registry import get_schema_registry
except ImportError:
    get_schema_registry = None

logger = logging.getLogger(__name__)


def infer_entity_class(name: str) -> str:
    """Infer entity class from name using keyword matching."""
    if not name:
        return "Concept"

    name_lower = name.lower()

    system_keywords = ["system", "platform", "application", "app", "software", "tool",
                       "service", "solution", "ERP", "PLM", "CRM", "SCM", "SAP",
                       "database", "server", "API", "systemweaver", "bis"]
    capability_keywords = ["management", "capability", "process", "function",
                           "service", "ability", "operation", "workflow", "control"]
    actor_keywords = ["team", "department", "role", "user", "engineer", "supplier",
                      "customer", "manager", "operator", "agent", "person"]
    asset_keywords = ["factory", "plant", "warehouse", "machine", "equipment",
                      "facility", "building", "vehicle", "hardware"]
    data_keywords = ["data", "record", "document", "file", "information",
                     "report", "bom", "vin", "spec", "specification"]
    process_keywords = ["process", "flow", "procedure", "workflow", "method",
                        "operation", "step", "cycle", "chain"]
    business_object_keywords = ["part", "product", "order", "item", "object",
                                "document", "record", "case"]
    storage_keywords = ["location", "storage", "warehouse", "repository",
                        "shelf", "bin", "datacenter", "dc"]

    for keywords, cls in [
        (system_keywords, "System"),
        (capability_keywords, "Capability"),
        (actor_keywords, "Actor"),
        (asset_keywords, "Asset"),
        (data_keywords, "Data"),
        (process_keywords, "Process"),
        (business_object_keywords, "BusinessObject"),
        (storage_keywords, "StorageLocation"),
    ]:
        for keyword in keywords:
            if keyword in name_lower:
                return cls

    return "Concept"


def validate_extraction(
    l1_entities: List[Dict],
    l2_entities: List[Dict],
    l25_entities: List[Dict],
    relationships: List[Dict],
    use_pydantic: bool = True
) -> Dict[str, Any]:
    """Validate extraction result using JSON Schema and Pydantic.

    Phase 1 Improvement: Dual validation with both legacy JSON Schema
    and new Pydantic models for maximum robustness.

    Args:
        l1_entities: L1 extracted entities
        l2_entities: L2 extracted entities
        l25_entities: L2.5 extracted entities
        relationships: Extracted relationships
        use_pydantic: Whether to use Pydantic validation (default: True)

    Returns:
        Validation report with 'valid', 'errors', 'warnings', 'stats'
    """
    report = {"valid": True, "errors": [], "warnings": [], "stats": {}}

    # Phase 1: Pydantic structured validation (new)
    if use_pydantic:
        try:
            from ontology_compiler.schemas.unified_extraction_models import (
                UnifiedExtractionData, ExtractionValidationResult, SCHEMA_VERSION
            )

            data = {
                "l1_entities": l1_entities,
                "l2_entities": l2_entities,
                "l25_entities": l25_entities,
                "relationships": relationships
            }

            # Normalize attribute payloads before strict model validation so that
            # common LLM shorthand (e.g. string lists) can be repaired into the
            # canonical EntityAttribute[] representation.
            for entity in data.get("l1_entities", []):
                if isinstance(entity, dict):
                    attrs = entity.get("attributes")
                    if isinstance(attrs, list):
                        repaired_attrs = []
                        for item in attrs:
                            if isinstance(item, str):
                                repaired_attrs.append({"name": item, "value": ""})
                            elif isinstance(item, dict):
                                if "name" in item or "key" in item or "attribute" in item:
                                    repaired_attrs.append(item)
                                elif len(item) == 1:
                                    key, value = next(iter(item.items()))
                                    repaired_attrs.append({"name": str(key), "value": "" if value is None else str(value)})
                                else:
                                    repaired_attrs.append(item)
                            else:
                                repaired_attrs.append({"name": str(item), "value": ""})
                        entity["attributes"] = repaired_attrs
                    elif isinstance(attrs, dict):
                        entity["attributes"] = [
                            {"name": str(k), "value": "" if v is None else str(v)}
                            for k, v in attrs.items()
                        ]

            # Attempt Pydantic validation
            extraction_data = UnifiedExtractionData.model_validate(data)
            pydantic_stats = extraction_data.get_stats()
            report["stats"] = pydantic_stats
            report["schema_version"] = SCHEMA_VERSION

            # Check for empty results (warning, not error)
            if pydantic_stats["l1_count"] == 0:
                report["warnings"].append("No L1 entities extracted")
            if pydantic_stats["l2_count"] == 0:
                report["warnings"].append("No L2 entities extracted")
            if pydantic_stats["l25_count"] == 0:
                report["warnings"].append("No L2.5 entities extracted")
            if pydantic_stats["relationship_count"] == 0:
                report["warnings"].append("No relationships extracted")

            logger.debug(f"Pydantic validation passed: {pydantic_stats}")

        except Exception as e:
            # Pydantic validation failed - collect detailed errors
            error_msg = f"Pydantic validation failed: {e}"
            logger.warning(error_msg)
            report["errors"].append(error_msg)
            report["valid"] = False

            # Attempt to provide specific field-level errors
            if hasattr(e, 'errors'):
                for err in e.errors():
                    loc = ".".join(str(x) for x in err.get('loc', []))
                    msg = err.get('msg', 'Unknown error')
                    report["errors"].append(f"  {loc}: {msg}")

            # [FIX] Always compute stats even when validation fails
            # This allows degraded mode to proceed with partial results
            report["stats"] = {
                "l1_count": len(l1_entities),
                "l2_count": len(l2_entities),
                "l25_count": len(l25_entities),
                "relationship_count": len(relationships)
            }

    # Phase 1: Legacy JSON Schema validation (fallback)
    try:
        from ontology_compiler.schemas.extraction_schema import validate_extraction_result

        data = {
            "l1_entities": l1_entities,
            "l2_entities": l2_entities,
            "l25_entities": l25_entities,
            "relationships": relationships
        }

        legacy_result = validate_extraction_result(data, strict=False)

        # Merge legacy results
        if not legacy_result.get("valid", True):
            report["valid"] = False
            report["errors"].extend(legacy_result.get("errors", []))
        report["warnings"].extend(legacy_result.get("warnings", []))

    except ImportError:
        report["warnings"].append("Legacy schema validation skipped (ImportError)")
    except Exception as e:
        logger.warning(f"Legacy validation error: {e}")
        report["warnings"].append(f"Legacy validation error: {e}")

    # Phase 2: SchemaRegistry validation (R5 - enforce schema compliance)
    try:
        registry = get_schema_registry()

        all_entities = l1_entities + l2_entities + l25_entities
        invalid_entities = []
        invalid_relations = []

        def is_valid_class(meta_class: str) -> bool:
            if registry.is_valid_main_class(meta_class):
                return True
            for mc in registry.main_classes:
                if registry.is_valid_subclass(mc, meta_class):
                    return True
            return False

        for entity in all_entities:
            meta_class = entity.get("meta_class", "").strip()
            if meta_class and not is_valid_class(meta_class):
                invalid_entities.append({
                    "meta_id": entity.get("meta_id"),
                    "name": entity.get("name") or entity.get("scenario_name"),
                    "meta_class": meta_class,
                    "layer": "L1" if entity in l1_entities else ("L2" if entity in l2_entities else "L2.5")
                })

        for rel in relationships:
            rel_type = rel.get("type", "").strip()
            if rel_type and not registry.is_valid_relation_type(rel_type):
                invalid_relations.append({
                    "from": rel.get("from"),
                    "to": rel.get("to"),
                    "type": rel_type
                })

        if invalid_entities:
            for invalid in invalid_entities:
                error_msg = (
                    f"[SCHEMA] Invalid meta_class '{invalid['meta_class']}' "
                    f"for entity '{invalid['name'] or invalid['meta_id']}' "
                    f"(layer: {invalid['layer']})"
                )
                report["errors"].append(error_msg)
                logger.warning(error_msg)
            report["valid"] = False

        if invalid_relations:
            for invalid in invalid_relations:
                warn_msg = (
                    f"[SCHEMA] Invalid relation_type '{invalid['type']}' "
                    f"for relation {invalid['from']} -> {invalid['to']}, will default to 'associated_with'"
                )
                report["warnings"].append(warn_msg)
                logger.info(warn_msg)

        if invalid_entities or invalid_relations:
            report["warnings"].append(
                f"[SCHEMA] {len(invalid_entities)} entities and {len(invalid_relations)} "
                f"relationships failed SchemaRegistry validation"
            )

        report["schema_registry_version"] = registry.version

    except ImportError:
        report["warnings"].append("SchemaRegistry validation skipped (ImportError)")
    except Exception as e:
        logger.warning(f"SchemaRegistry validation error: {e}")
        report["warnings"].append(f"SchemaRegistry validation error: {e}")

    return report


def validate_response_semantics(
    l1_entities: List[Dict],
    l2_entities: List[Dict],
    l25_entities: List[Dict],
    l3_entities: List[Dict],
    relationships: List[Dict],
) -> Dict[str, Any]:
    """Validate semantic contract of extracted entities and relationships."""
    errors = []
    warnings = []
    all_entities = l1_entities + l2_entities + l25_entities + l3_entities
    entity_ids = {str(e.get("meta_id", "")).strip() for e in all_entities if isinstance(e, dict)}

    for entity in all_entities:
        if not isinstance(entity, dict):
            continue
        meta_id = str(entity.get("meta_id", "")).strip()
        meta_class = str(entity.get("meta_class", "")).strip()
        layer = str(entity.get("layer", "")).strip()
        evidence = entity.get("evidence")
        if not meta_id:
            errors.append({"type": "missing_meta_id", "entity": entity})
        if not meta_class:
            errors.append({"type": "missing_meta_class", "entity": entity})
        if not evidence:
            warnings.append({"type": "missing_evidence", "entity": entity})
        if layer == "L3" and meta_class != "Instance":
            warnings.append({"type": "l3_not_instance", "entity": entity})

    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        source = str(rel.get("from") or rel.get("source") or rel.get("source_id") or "").strip()
        target = str(rel.get("to") or rel.get("target") or rel.get("target_id") or "").strip()
        rel_type = str(rel.get("type") or rel.get("rel_type") or "").strip()
        if not source or not target:
            errors.append({"type": "relationship_missing_endpoint", "relationship": rel})
        if rel_type and rel_type not in (get_schema_registry().relation_types if get_schema_registry else [rel_type]):
            warnings.append({"type": "relationship_type_invalid", "relationship": rel})
        if source and source not in entity_ids:
            warnings.append({"type": "relationship_source_unknown", "relationship": rel})
        if target and target not in entity_ids:
            warnings.append({"type": "relationship_target_unknown", "relationship": rel})

    return {"valid": not errors, "errors": errors, "warnings": warnings}


def add_validation_feedback(
    prompt: str,
    validation_result: Dict[str, Any]
) -> str:
    """Add validation feedback to prompt for retry.

    Args:
        prompt: Original prompt
        validation_result: Validation errors

    Returns:
        Enhanced prompt with validation feedback
    """
    errors = validation_result.get("errors", [])
    if not errors:
        return prompt

    feedback = "\n\n## PREVIOUS ATTEMPT FEEDBACK\n"
    feedback += "The previous extraction had the following issues:\n"
    for error in errors:
        feedback += f"- {error}\n"
    feedback += "\nPlease fix these issues in your response.\n"

    return prompt + feedback


def add_governance_feedback(
    prompt: str,
    governance_report: Dict[str, Any]
) -> str:
    """Add skill governance feedback to prompt for retry.

    Args:
        prompt: Original prompt
        governance_report: Governance validation result

    Returns:
        Enhanced prompt with governance feedback
    """
    issues = governance_report.get("issues", [])
    if not issues:
        return prompt

    feedback = "\n\n## GOVERNANCE VIOLATION FEEDBACK\n"
    feedback += "The previous extraction had the following governance violations:\n\n"

    blocked_relations = governance_report.get("blocked_relations", 0)
    entity_violations = governance_report.get("entity_type_violations", 0)
    field_violations = governance_report.get("required_field_violations", 0)

    if blocked_relations > 0:
        feedback += "### BLOCKED RELATIONSHIPS\n"
        feedback += f"- {blocked_relations} relationships use invalid relation types\n"
        feedback += "- Valid relation types: contains, implements, depends_on, uses, associated_with, calls, produces, consumes, triggers, conflicts_with, realizes, maintains, manages, deploys_to, executes, governs_data, results_in, creates, provides_data_for, enhances, defines, maps_to\n"

    if entity_violations > 0:
        feedback += "\n### ENTITY TYPE VIOLATIONS\n"
        feedback += f"- {entity_violations} entities have invalid meta_class or meta_subclass values\n"

    if field_violations > 0:
        feedback += "\n### MISSING REQUIRED FIELDS\n"
        feedback += f"- {field_violations} entities are missing required fields\n"

    feedback += "\n### SPECIFIC VIOLATIONS\n"
    for i, issue in enumerate(issues[:10], 1):
        detail = issue.get("detail", "")
        skill_id = issue.get("skill_id", "")
        issue_type = issue.get("type", "")
        feedback += f"{i}. [{issue_type}] {detail} (skill: {skill_id})\n"

    feedback += "\nPlease correct these issues in your response:\n"
    feedback += "- Use ONLY valid relation types from the list above\n"
    feedback += "- Ensure all entities have valid meta_class values\n"
    feedback += "- Complete all required fields for each entity\n"

    return prompt + feedback


def validate_with_schema_registry(
    l1_entities: List[Dict],
    l2_entities: List[Dict],
    l25_entities: List[Dict],
    relationships: List[Dict],
    domain: Optional[str] = None
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Validate meta_class and meta_subclass against SchemaRegistry and filter invalid entities.

    Args:
        l1_entities: L1 entities to validate
        l2_entities: L2 entities to validate
        l25_entities: L2.5 entities to validate
        relationships: Relationships to validate
        domain: Optional domain hint

    Returns:
        Tuple of (l1_entities, l2_entities, l25_entities, relationships) with invalid items filtered
    """
    registry = get_schema_registry()
    if not registry:
        logger.warning("SchemaRegistry not available - skipping schema validation")
        return l1_entities, l2_entities, l25_entities, relationships

    valid_l1 = []
    valid_l2 = []
    valid_l25 = []
    valid_relationships = []

    invalid_count = 0

    VALID_L0_CLASSES = set(registry.main_classes)

    def _validate_entity(entity: Dict, layer: str, default_class: str) -> bool:
        nonlocal invalid_count
        meta_class_raw = entity.get("meta_class", "")
        meta_class = str(meta_class_raw).strip() if meta_class_raw is not None else ""
        meta_subclass_raw = entity.get("meta_subclass", "")
        meta_subclass = str(meta_subclass_raw).strip() if meta_subclass_raw is not None else ""

        if not meta_class:
            meta_class = default_class

        normalized_class = registry.normalize_main_class(meta_class)

        is_valid = registry.is_valid_main_class(normalized_class)
        if not is_valid:
            for mc in registry.main_classes:
                if registry.is_valid_subclass(mc, meta_class):
                    normalized_class = mc
                    if not meta_subclass:
                        meta_subclass = meta_class
                    is_valid = True
                    break

        if not is_valid:
            normalized_class, auto_subclass = _auto_map_to_main_class(meta_class, layer)
            if auto_subclass and not meta_subclass:
                meta_subclass = auto_subclass
            logger.warning(
                f"[SCHEMA] Unknown meta_class '{meta_class}', auto-mapped to '{normalized_class}' "
                f"for entity '{entity.get('name') or entity.get('scenario_name') or entity.get('meta_id')}' "
                f"(layer: {layer})"
            )
            invalid_count += 1

        entity["meta_class"] = normalized_class

        if meta_subclass:
            normalized_subclass = registry.normalize_subclass(meta_subclass, normalized_class)
            if not registry.is_valid_subclass(normalized_class, normalized_subclass):
                entity["meta_subclass"] = meta_subclass
                logger.warning(
                    f"[SCHEMA] Unknown meta_subclass '{meta_subclass}' preserved as-is "
                    f"for meta_class '{normalized_class}' in entity '{entity.get('name') or entity.get('scenario_name') or entity.get('meta_id')}' "
                    f"(layer: {layer})"
                )
                invalid_count += 1
            else:
                entity["meta_subclass"] = normalized_subclass
        else:
            entity["meta_subclass"] = ""

        return True

    def _auto_map_to_main_class(meta_class: str, layer: str) -> Tuple[str, str]:
        lower_class = meta_class.lower()
        subclass = meta_class

        if any(keyword in lower_class for keyword in ["system", "platform", "server", "service", "app", "software", "deployment"]):
            return "System", subclass
        elif any(keyword in lower_class for keyword in ["actor", "role", "user", "department", "team", "function"]):
            return "Actor", subclass
        elif any(keyword in lower_class for keyword in ["asset", "component", "vehicle", "device", "part"]):
            return "Asset", subclass
        elif any(keyword in lower_class for keyword in ["businessrule", "validationrule", "mappingrule", "transformrule", "rule", "constraint", "condition"]):
            return "Data", subclass
        elif any(keyword in lower_class for keyword in ["data", "artifact", "document", "record", "repository"]):
            return "Data", subclass
        elif any(keyword in lower_class for keyword in ["process", "flow", "workflow", "procedure"]):
            return "Process", subclass
        elif any(keyword in lower_class for keyword in ["capability", "domain", "feature", "ability"]):
            return "Capability", subclass

        if layer == "L1":
            return "System", subclass
        elif layer == "L2":
            return "System", subclass
        else:
            return "Data", subclass

    for entity in l1_entities:
        if isinstance(entity, dict) and _validate_entity(entity, "L1", "System"):
            valid_l1.append(entity)

    for entity in l2_entities:
        if isinstance(entity, dict) and _validate_entity(entity, "L2", "System"):
            valid_l2.append(entity)

    for entity in l25_entities:
        if isinstance(entity, dict) and _validate_entity(entity, "L2.5", "Data"):
            valid_l25.append(entity)

    for rel in relationships:
        if isinstance(rel, dict):
            rel_type_raw = rel.get("rel_type") if rel.get("rel_type") is not None else rel.get("type")
            rel_type = str(rel_type_raw).strip() if rel_type_raw is not None else ""
            normalized_type = registry.normalize_relation_type(rel_type)
            if registry.is_valid_relation_type(normalized_type):
                rel["type"] = normalized_type
                valid_relationships.append(rel)
            elif rel_type:
                rel["type"] = "associated_with"
                valid_relationships.append(rel)
                logger.warning(
                    f"[SCHEMA] Unknown relation_type '{rel_type}' "
                    f"for relation {rel.get('from') or rel.get('source')} -> {rel.get('to') or rel.get('target')}, "
                    f"defaulting to 'associated_with'"
                )
                invalid_count += 1

    if invalid_count > 0:
        logger.info(
            f"[SCHEMA] Validation complete: {invalid_count} invalid items filtered. "
            f"Result: L1={len(valid_l1)}, L2={len(valid_l2)}, L2.5={len(valid_l25)}, Rels={len(valid_relationships)}"
        )

    return valid_l1, valid_l2, valid_l25, valid_relationships
