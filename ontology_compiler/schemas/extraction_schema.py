"""JSON Schema for UnifiedExtractionResult validation.

Provides structured validation for LLM extraction output.
"""

from __future__ import annotations

EXTRACTION_SCHEMA = {
    "type": "object",
    "required": ["l1_entities", "l2_entities", "l25_entities", "relationships"],
    "properties": {
        "l1_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["meta_id", "meta_name", "meta_class"],
                "properties": {
                    "meta_id": {"type": "string", "minLength": 1},
                    "meta_name": {"type": "string", "minLength": 1},
                    "meta_class": {"type": "string", "minLength": 1},
                    "meta_subclass": {"type": "string"},
                    "description": {"type": "string"},
                    "domain": {"type": "string"},
                }
            }
        },
        "l2_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["meta_id", "scenario_name"],
                "properties": {
                    "meta_id": {"type": "string", "minLength": 1},
                    "scenario_name": {"type": "string", "minLength": 1},
                    "domain": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "participants": {"type": "array", "items": {"type": "string"}},
                    "triggers": {"type": "array", "items": {"type": "string"}},
                    "outcomes": {"type": "array", "items": {"type": "string"}},
                }
            }
        },
        "l25_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "meta_id": {"type": "string"},
                    "rule_name": {"type": "string"},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "transformation": {"type": "string"},
                    "description": {"type": "string"},
                }
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to", "type"],
                "properties": {
                    "from": {"type": "string", "minLength": 1},
                    "to": {"type": "string", "minLength": 1},
                    "source": {"type": "string", "minLength": 1},
                    "target": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "minLength": 1},
                    "layer": {"type": "string"},
                    "description": {"type": "string"},
                }
            }
        }
    }
}


def validate_extraction_result(data: dict, strict: bool = False, document_type: str | None = None) -> dict:
    """Validate extraction result against schema.
    
    Args:
        data: Extraction result data
        strict: If True, require all fields; if False, allow partial results
        
    Returns:
        Validation report with 'valid', 'errors', 'warnings'
    """
    report = {"valid": True, "errors": [], "warnings": []}
    
    # Check required top-level keys
    for key in EXTRACTION_SCHEMA["required"]:
        if key not in data:
            report["errors"].append(f"Missing required key: {key}")
            report["valid"] = False
    
    if not report["valid"] and strict:
        if document_type:
            dtype = document_type.lower()

            def _contains(items, terms):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    text = " ".join(str(item.get(k, "")) for k in ("meta_name", "name", "description", "relation_type", "type", "scenario_name")).lower()
                    if any(term in text for term in terms):
                        return True
                return False

            if dtype == "mbom":
                for label, terms in {
                    "position": ["position"],
                    "variant": ["variant"],
                    "occurrence": ["occurrence"],
                    "provenance": ["provenance"],
                    "source_structure": ["source structure", "source_structure", "structure_level"],
                    "alignment": ["align", "alignment"],
                }.items():
                    if not _contains(data.get("l1_entities", []) + data.get("l2_entities", []) + data.get("l25_entities", []), terms):
                        report["errors"].append(f"MBOM gate missing {label}")
                        report["valid"] = False
            elif dtype == "benchmark":
                for label, terms in {"scope": ["scope"], "kpi": ["kpi"], "question": ["question"]}.items():
                    if not _contains(data.get("l1_entities", []) + data.get("l2_entities", []) + data.get("l25_entities", []), terms):
                        report["errors"].append(f"Benchmark gate missing {label}")
                        report["valid"] = False
            elif dtype == "process_governance":
                for label, terms in {"rule": ["rule"], "workflow": ["workflow", "process"], "owner": ["owner", "approver"]}.items():
                    if not _contains(data.get("l1_entities", []) + data.get("l2_entities", []) + data.get("l25_entities", []), terms):
                        report["errors"].append(f"Governance gate missing {label}")
                        report["valid"] = False

    for key in ["l1_entities", "l2_entities", "l25_entities", "relationships"]:
        items = data.get(key, [])
        if not isinstance(items, list):
            report["errors"].append(f"{key} must be an array")
            report["valid"] = False
            continue
        
        if len(items) == 0:
            report["warnings"].append(f"{key} is empty")
        
        schema = EXTRACTION_SCHEMA["properties"].get(key, {})
        required = schema.get("items", {}).get("required", [])
        
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                report["errors"].append(f"{key}[{i}] must be an object")
                report["valid"] = False
                continue
            
            for field in required:
                if field not in item:
                    report["errors"].append(f"{key}[{i}] missing required field: {field}")
                    report["valid"] = False
    
    return report
