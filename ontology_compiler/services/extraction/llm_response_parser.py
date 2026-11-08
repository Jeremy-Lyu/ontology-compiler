"""LLM response parsing and JSON repair for the unified extractor.

Pure helpers extracted from ``true_unified_extractor`` so the extractor class can
act as an orchestrator instead of owning the parsing pipeline. These functions
carry no instance state and keep the exact behavior of the original methods.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from core.utils.json_parser import parse_llm_output

logger = logging.getLogger(__name__)

try:
    from ontology_compiler.services.failure_memory import record_json_parse_failure
except ImportError:  # pragma: no cover - optional observability hook
    def record_json_parse_failure(*args, **kwargs):
        pass


_EMPTY_EXTRACTION = {
    "L0_entities": [],
    "L1_entities": [],
    "L2_entities": [],
    "L2_5_entities": [],
    "L3_entities": [],
    "relationships": [],
}

_VALID_META_CLASSES = {"Capability", "System", "Actor", "Asset", "Data", "Process"}

_REQUIRED_ENTITY_FIELDS = ["meta_id", "meta_class", "name", "evidence"]

_LAYER_ENTITIES = [
    ("L0", "L0_entities"),
    ("L1", "L1_entities"),
    ("L2", "L2_entities"),
    ("L2.5", "L2_5_entities"),
    ("L3", "L3_entities"),
]


def clean_llm_response(response: str) -> str:
    """Strip think tags and markdown fences from a raw LLM response."""
    cleaned = response.strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned)
    cleaned = re.sub(r"^\s*text\s*=\s*", "", cleaned)
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Attempt to extract JSON from text response using the unified parser."""
    result = parse_llm_output(text)
    if result.success:
        return result.data

    logger.warning(f"LLM output parse failed: {result.error_message}")

    fallback = repair_json(text)
    if any(fallback.values()):
        return fallback

    return dict(_EMPTY_EXTRACTION)


def parse_unified_response(
    response: str,
    infer_entity_class: Any,
    assign_meta_ids: Any,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Parse a unified LLM response into per-layer entity lists plus relationships.

    ``infer_entity_class`` and ``assign_meta_ids`` are callables owned by the
    caller so this function stays free of instance state.
    """
    cleaned = clean_llm_response(response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Response preview: {cleaned[:500]}...")
        record_json_parse_failure(
            error_message=f"JSON parse error: {e}",
            document_type="llm_response",
            layer="L1/L2/L2.5",
        )
        data = extract_json_from_text(cleaned)

    l0_entities: List[Dict] = []
    l1_entities: List[Dict] = []
    l2_entities: List[Dict] = []
    l25_entities: List[Dict] = []
    l3_entities: List[Dict] = []
    relationships: List[Dict] = []

    if isinstance(data, dict):
        l0_entities = data.get("L0_entities") or data.get("l0_entities") or data.get("L0") or []
        l1_entities = data.get("L1_entities") or data.get("l1_entities") or data.get("L1") or []
        l2_entities = (
            data.get("L2_entities")
            or data.get("l2_entities")
            or data.get("L2")
            or data.get("business_scenarios")
            or []
        )
        l25_entities = (
            data.get("L2_5_entities")
            or data.get("L25_entities")
            or data.get("l25_entities")
            or data.get("L2.5")
            or data.get("mapping_rules")
            or []
        )
        l3_entities = (
            data.get("L3_entities") or data.get("l3_entities") or data.get("L3") or data.get("instances") or []
        )
        relationships = data.get("relationships") or data.get("RELATIONSHIPS") or []
        for ent_list, layer in _LAYER_ENTITIES:
            if isinstance(ent_list, list):
                for entity in ent_list:
                    if isinstance(entity, dict):
                        entity.setdefault("layer", layer)
                        entity.setdefault("raw_fields", {})

        if "entities" in data and isinstance(data["entities"], dict):
            entities = data["entities"]
            l0_entities = entities.get("L0") or l0_entities
            l1_entities = entities.get("L1") or l1_entities
            l2_entities = entities.get("L2") or l2_entities
            l25_entities = entities.get("L2.5") or l25_entities
            l3_entities = entities.get("L3") or l3_entities
            relationships = entities.get("relationships") or relationships

    l0_entities = l0_entities if isinstance(l0_entities, list) else []
    l1_entities = l1_entities if isinstance(l1_entities, list) else []
    l2_entities = l2_entities if isinstance(l2_entities, list) else []
    l25_entities = l25_entities if isinstance(l25_entities, list) else []
    l3_entities = l3_entities if isinstance(l3_entities, list) else []
    relationships = relationships if isinstance(relationships, list) else []

    for entities, layer in _LAYER_ENTITIES:
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                continue
            missing = [f for f in _REQUIRED_ENTITY_FIELDS if f not in entity or not entity[f]]
            if missing:
                logger.warning(f"{layer} entity {i} missing required fields: {missing}")
                for field in missing:
                    if field == "meta_class":
                        entity[field] = infer_entity_class(entity.get("name", ""))
                    elif field == "evidence":
                        entity[field] = {"content": "Extracted from document content", "confidence": 0.7}
                    elif field == "meta_id":
                        entity[field] = f"{layer}-{i}"
                entity.setdefault("normalization_flags", [])
                entity["normalization_flags"].append("schema_autofill")

    for i, rel in enumerate(relationships):
        if not isinstance(rel, dict):
            continue
        missing = []
        if "from" not in rel or not rel["from"]:
            missing.append("from")
        if "to" not in rel or not rel["to"]:
            missing.append("to")
        if "type" not in rel or not rel["type"]:
            missing.append("type")
        if missing:
            logger.warning(f"Relationship {i} missing required fields: {missing}")
        if "confidence" not in rel or rel["confidence"] == 0:
            rel["confidence"] = 0.7

    for entity in l0_entities:
        if isinstance(entity, dict):
            if "meta_name" not in entity and "name" in entity:
                entity["meta_name"] = entity["name"]
            if "meta_class" not in entity:
                entity["meta_class"] = "Capability"
            if "evidence" not in entity or not entity["evidence"]:
                entity["evidence"] = {"content": "Extracted from document content", "confidence": 0.7}

    for entity in l1_entities:
        if isinstance(entity, dict):
            if "meta_name" not in entity and "name" in entity:
                entity["meta_name"] = entity["name"]
            if "parent_class" not in entity:
                entity["parent_class"] = "Capability"
            if "evidence" not in entity or not entity["evidence"]:
                entity["evidence"] = {"content": "Extracted from document content", "confidence": 0.7}

    for entity in l2_entities:
        if isinstance(entity, dict):
            if "scenario_name" not in entity and "name" in entity:
                entity["scenario_name"] = entity["name"]
            if "evidence" not in entity or not entity["evidence"]:
                entity["evidence"] = {"content": "Extracted from document content", "confidence": 0.7}

    for entity in l25_entities:
        if isinstance(entity, dict):
            if "meta_id" not in entity and "name" in entity:
                entity["meta_id"] = f"L25-{entity['name'][:20].replace(' ', '_')}"
            if "evidence" not in entity or not entity["evidence"]:
                entity["evidence"] = {"content": "Extracted from document content", "confidence": 0.7}

    for entity in l3_entities:
        if isinstance(entity, dict):
            if "meta_name" not in entity and "name" in entity:
                entity["meta_name"] = entity["name"]
            meta_class = entity.get("meta_class", "")
            if not meta_class or meta_class not in _VALID_META_CLASSES:
                entity["meta_class"] = "Asset"
            if "evidence" not in entity or not entity["evidence"]:
                entity["evidence"] = {"content": "Extracted from document content", "confidence": 0.7}

    l0_entities = assign_meta_ids(l0_entities, "L0")
    l1_entities = assign_meta_ids(l1_entities, "L1")
    l2_entities = assign_meta_ids(l2_entities, "L2")
    l25_entities = assign_meta_ids(l25_entities, "L25")
    l3_entities = assign_meta_ids(l3_entities, "L3")

    return l0_entities, l1_entities, l2_entities, l25_entities, l3_entities, relationships


def _balance_braces(text: str) -> str:
    """Balance unclosed braces and brackets."""
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1

    result = text
    if brace_count > 0:
        result += "}" * brace_count
    if bracket_count > 0:
        result += "]" * bracket_count

    return result


def _fix_missing_commas(text: str) -> str:
    """Fix missing commas between JSON elements."""
    text = re.sub(r'"\}\s*"', '"}, "', text)
    text = re.sub(r"]\s*\{", "], {", text)
    text = re.sub(r"\}\s*\{", "}, {", text)
    text = re.sub(r"\}\s*\[", "}, [", text)
    text = re.sub(r"]\s*\[", "], [", text)
    text = re.sub(r'"\s*\}', '", }', text)
    text = re.sub(r'"\s*\[', '", [', text)
    text = re.sub(r'"\s*\{', '", {', text)
    text = re.sub(r']\s*"', '], "', text)
    text = re.sub(r'\}\s*"', '}, "', text)
    return text


def _fix_unclosed_brackets(text: str) -> str:
    """Close unclosed brackets/braces at end of JSON."""
    return _balance_braces(text)


def _remove_incomplete_elements(text: str) -> str:
    """Remove incomplete elements at the end of arrays/objects."""
    text = re.sub(r',\s*"[^"]*$', "", text)
    text = re.sub(r',\s*"[\w_]+":\s*[^,}\]]*$', "", text)
    return text


def _fix_comma_delimiter_errors(text: str) -> str:
    """Fix 'Expecting ',' delimiter' errors by detecting common JSON formatting issues."""
    text = re.sub(r'"(.*?)"\s+(?=")', r'"\1", ', text)
    text = re.sub(r'"(.*?)"\s+(?=\{)', r'"\1", ', text)
    text = re.sub(r'"(.*?)"\s+(?=\[)', r'"\1", ', text)
    text = re.sub(r'"(.*?)"\s+(?=\})', r'"\1", ', text)
    text = re.sub(r'"(.*?)"\s+(?=\])', r'"\1", ', text)
    text = re.sub(r'(\d+)\s+(?=")', r'\1, ', text)
    text = re.sub(r'(\d+)\s+(?=\{)', r'\1, ', text)
    text = re.sub(r'(\d+)\s+(?=\[)', r'\1, ', text)
    text = re.sub(r'(\d+)\s+(?=\})', r'\1, ', text)
    text = re.sub(r'(\d+)\s+(?=\])', r'\1, ', text)
    text = re.sub(r']\s+(?=")', '], ', text)
    text = re.sub(r']\s+(?=\{)', '], ', text)
    text = re.sub(r'\}\s+(?=")', '}, ', text)
    text = re.sub(r'\}\s+(?=\{)', '}, ', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r',\s*\}', '}', text)
    text = re.sub(r',\s*\]', ']', text)
    return text


def _extract_complete_elements(text: str) -> list:
    """Extract complete JSON elements from potentially truncated array content."""
    elements = []
    current_element = ""
    brace_level = 0
    bracket_level = 0
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            current_element += char
            escape_next = False
            continue
        if char == "\\":
            current_element += char
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            current_element += char
            continue
        if in_string:
            current_element += char
            continue
        if char == "{":
            brace_level += 1
            current_element += char
        elif char == "}":
            brace_level -= 1
            current_element += char
            if brace_level == 0 and bracket_level == 0:
                try:
                    elements.append(json.loads(current_element))
                except Exception:
                    pass
                current_element = ""
        elif char == "[":
            bracket_level += 1
            current_element += char
        elif char == "]":
            bracket_level -= 1
            current_element += char
            if brace_level == 0 and bracket_level == 0:
                try:
                    elements.append(json.loads(current_element))
                except Exception:
                    pass
                current_element = ""
        elif char == ",":
            if brace_level == 0 and bracket_level == 0:
                if current_element.strip():
                    try:
                        elements.append(json.loads(current_element.strip()))
                    except Exception:
                        pass
                current_element = ""
            else:
                current_element += char
        else:
            current_element += char

    return elements


def _extract_partial_json(text: str) -> Dict[str, Any]:
    """Extract partial JSON data by parsing known field names."""
    result = dict(_EMPTY_EXTRACTION)
    for key in _EMPTY_EXTRACTION:
        match = re.search(rf'"{key}"\s*:', text)
        if match:
            after_key = text[match.end():]
            bracket_match = re.match(r"\s*\[", after_key)
            if bracket_match:
                array_start = match.end() + bracket_match.end()
                result[key] = _extract_complete_elements(text[array_start:])
    return result


def _aggressive_json_repair(text: str) -> Dict[str, Any]:
    """Aggressively repair severely malformed JSON by extracting valid entities line by line."""
    result = dict(_EMPTY_EXTRACTION)

    for layer_key in ["L0_entities", "L1_entities", "L2_entities", "L2_5_entities", "L3_entities"]:
        pattern = r'\{[^{}]*"meta_id"\s*:\s*"[^"]*"[^{}]*\}'
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                obj = json.loads(match)
                if "meta_id" in obj:
                    result[layer_key].append(obj)
            except Exception:
                try:
                    fixed = re.sub(r",\s*}", "}", match)
                    fixed = re.sub(r",\s*]", "]", fixed)
                    obj = json.loads(fixed)
                    if "meta_id" in obj:
                        result[layer_key].append(obj)
                except Exception:
                    continue

    rel_pattern = r'\{[^{}]*"(?:from|source)"\s*:\s*"[^"]*"[^{}]*\}'
    rel_matches = re.findall(rel_pattern, text)
    for match in rel_matches:
        try:
            result["relationships"].append(json.loads(match))
        except Exception:
            continue

    total_extracted = sum(len(v) for v in result.values())
    if total_extracted > 0:
        logger.info(f"Aggressive JSON repair extracted {total_extracted} items")

    return result


def _escape_internal_quotes(text: str) -> str:
    """Escape unescaped quotes within JSON strings."""
    result = []
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            result.append(char)
            escape_next = False
            continue
        if char == "\\":
            result.append(char)
            escape_next = True
            continue
        if char == '"':
            if in_string:
                if i + 1 < len(text):
                    next_char = text[i + 1]
                    if next_char in ',}]":':
                        in_string = False
                        result.append(char)
                    else:
                        result.append("\\")
                        result.append(char)
                else:
                    in_string = False
                    result.append(char)
            else:
                in_string = True
                result.append(char)
        else:
            result.append(char)

    return "".join(result)


def repair_json(text: str) -> Dict[str, Any]:
    """Robust JSON repair mechanism for LLM responses."""
    original_text = text

    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", text)
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text)
    text = re.sub(r"<!--[\s\S]*?-->", "", text)

    text = re.sub(r"^```(json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    text = text.strip()
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        text = text[first_brace:]
    elif first_bracket != -1:
        text = text[first_bracket:]

    last_brace = text.rfind("}")
    last_bracket = text.rfind("]")
    if last_brace != -1 and last_brace > last_bracket:
        text = text[: last_brace + 1]
    elif last_bracket != -1:
        text = text[: last_bracket + 1]

    text = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:", r'"\1":', text)

    text = re.sub(r"}\s*\{", "}, {", text)
    text = re.sub(r"}\s*\[", "}, [", text)
    text = re.sub(r"]\s*\{", "], {", text)
    text = re.sub(r"]\s*\[", "], [", text)

    text = re.sub(r'"}\s*"', '"}, "', text)

    text = re.sub(r",\s*([}\]])", r"\1", text)

    text = _escape_internal_quotes(text)

    logger.debug(f"Repaired JSON: {text[:500]}...")

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct parse failed: {e}")

    try:
        return json.loads(_balance_braces(text))
    except Exception:
        pass

    try:
        return _extract_partial_json(text)
    except Exception:
        pass

    repair_attempts = [
        _fix_comma_delimiter_errors,
        _fix_missing_commas,
        _fix_unclosed_brackets,
        _remove_incomplete_elements,
    ]
    for repair_func in repair_attempts:
        try:
            repaired = repair_func(text)
            if repaired:
                return json.loads(repaired)
        except Exception:
            continue

    try:
        return _aggressive_json_repair(text)
    except Exception:
        pass

    return dict(_EMPTY_EXTRACTION)


__all__ = [
    "clean_llm_response",
    "parse_unified_response",
    "repair_json",
    "extract_json_from_text",
]
