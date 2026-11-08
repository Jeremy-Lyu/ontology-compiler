"""Entity normalization for the unified extractor.

Pure helpers extracted from ``true_unified_extractor`` so post-extraction
normalization (status defaults, naming conventions, attribute shaping) can be
tested and reused without instantiating the extractor.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_PREFIXES_TO_CAPITALIZE = {
    "API", "ERP", "MBOM", "SBOM", "PLM", "BOM", "KDP", "VSM",
    "RFID", "CD", "VCC", "CRM", "SCM", "WMS", "MES", "CAD", "CAE",
}

# Canonical legal status values — aligned with the ``status`` enum in
# ``schemas/ontology_packet.schema.json``:
#   ["active", "deprecated", "proposed", "draft", "ghost", "archived"]
# This is the single source of truth for status validity; both extract-stage
# post-processing and merge-stage validation must go through ``normalize_status``
# so the schema enum and runtime behavior cannot drift (R2 fact-source split).
_VALID_STATUSES = {"active", "deprecated", "proposed", "draft", "ghost", "archived"}

# Alias map for historically / LLM-produced values that are NOT in the canonical
# enum but appear in real extractions. Each maps to its closest canonical
# equivalent. This is the single place where status coercion happens.
#   planned  → proposed   (LLM frequently emits "planned"; schema enum has "proposed")
#   inactive → deprecated (schema has no "inactive"; closest semantic match)
#   unknown  → draft      (schema has no "unknown"; closest semantic match)
_STATUS_ALIASES = {
    "planned": "proposed",
    "inactive": "deprecated",
    "unknown": "draft",
}


def normalize_status(value: Any) -> str:
    """Coerce a status value to a canonical legal status.

    Single canonical entry point for status normalization. Order:
    1. Empty / None / literal ``"None"`` → ``"active"`` (default).
    2. Lowercase + alias map (e.g. ``planned`` → ``proposed``).
    3. If still not in :data:`_VALID_STATUSES` → ``"active"`` (safe default).

    Fixes the planned→proposed mapping at LLM extraction time: the schema
    rejects ``planned`` but allows ``proposed``, so LLM-emitted ``planned``
    is coerced here before it can reach schema validation.
    """
    if value is None or not isinstance(value, str) or value.strip() == "" or value == "None":
        return "active"
    normalized = value.strip().lower()
    normalized = _STATUS_ALIASES.get(normalized, normalized)
    if normalized not in _VALID_STATUSES:
        return "active"
    return normalized


_META_ID_PREFIX_MAP = {
    "L0": "L0-",
    "L1": "L1-",
    "L2": "L2-",
    "L2.5": "L25-",
    "L3": "L3-",
}


def _normalize_name(name: str) -> str:
    """Strip non-alphanumeric characters and apply title-case with acronym preservation."""
    if not name or not isinstance(name, str):
        return name

    name = name.strip()
    if not name:
        return name

    name = re.sub(r"[^\w\s\-]", "", name)

    words = name.split()
    normalized_words = []
    for word in words:
        if word.upper() in _PREFIXES_TO_CAPITALIZE:
            normalized_words.append(word.upper())
        elif word.isupper() and len(word) > 1:
            normalized_words.append(word)
        else:
            normalized_words.append(word.capitalize())

    return " ".join(normalized_words)


def _camelize_name(name: str) -> str:
    """Split camelCase names into capitalized words while preserving known acronyms."""
    if not name or not isinstance(name, str):
        return name

    name = name.strip()
    if not name:
        return name

    name = re.sub(r"[^\w\s\-]", "", name)

    words = re.split(r"[\s\-_]", name)
    if len(words) == 1:
        s = words[0]
        result = []
        for i, c in enumerate(s):
            if i == 0:
                result.append(c.upper())
            elif c.isupper() and i > 0 and s[i - 1].islower():
                result.append(" ")
                result.append(c)
            else:
                result.append(c)
        return "".join(result)

    normalized_words = []
    for word in words:
        if word.upper() in _PREFIXES_TO_CAPITALIZE:
            normalized_words.append(word.upper())
        elif word.isupper() and len(word) > 1:
            normalized_words.append(word)
        else:
            normalized_words.append(word.capitalize())

    return " ".join(normalized_words)


def _normalize_entity(entity: Dict[str, Any], expected_layer: str) -> Dict[str, Any]:
    """Normalize status, layer, meta_id prefix, names, and attribute shape."""
    if not isinstance(entity, dict):
        return entity

    # status: delegate to canonical normalizer (was: inline set check that
    # diverged from schemas/ontology_packet.schema.json — accepted "planned"
    # which the schema rejects, and rejected "proposed" which it allows).
    entity["status"] = normalize_status(entity.get("status"))

    if "layer" not in entity or not entity["layer"]:
        entity["layer"] = expected_layer

    expected_prefix = _META_ID_PREFIX_MAP.get(expected_layer, "L1-")

    meta_id = entity.get("meta_id", "")
    if meta_id and not (meta_id.startswith(expected_prefix) or meta_id.startswith(expected_prefix.replace("-", "_"))):
        if meta_id.startswith("L") and meta_id[1].isdigit():
            pass
        else:
            entity["meta_id"] = expected_prefix + meta_id

    if "name" in entity and isinstance(entity["name"], str):
        entity["name"] = _camelize_name(entity["name"])

    if "meta_name" in entity and isinstance(entity["meta_name"], str):
        entity["meta_name"] = _camelize_name(entity["meta_name"])

    if "scenario_name" in entity and isinstance(entity["scenario_name"], str):
        entity["scenario_name"] = _camelize_name(entity["scenario_name"])

    if "attributes" in entity:
        attrs = entity["attributes"]
        if isinstance(attrs, list):
            normalized_attrs = []
            for attr in attrs:
                if isinstance(attr, dict):
                    name = attr.get("name", "")
                    value = attr.get("value", "")
                    if name or value:
                        normalized_attrs.append(
                            {"name": str(name).strip(), "value": str(value).strip()}
                        )
                elif isinstance(attr, str):
                    parts = attr.split("=", 1)
                    if len(parts) == 2:
                        normalized_attrs.append(
                            {"name": parts[0].strip(), "value": parts[1].strip()}
                        )
                    elif attr.strip():
                        normalized_attrs.append({"name": attr.strip(), "value": ""})
            entity["attributes"] = normalized_attrs
        else:
            entity["attributes"] = []

    return entity


def normalize_entities(
    l0_entities: List[Dict[str, Any]],
    l1_entities: List[Dict[str, Any]],
    l2_entities: List[Dict[str, Any]],
    l25_entities: List[Dict[str, Any]],
    l3_entities: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], ...]:
    """Post-extraction normalization for all layers.

    Auto-fills missing status fields and normalizes entity names to comply with
    quality requirements regardless of LLM behavior.
    """
    normalized_l0 = [_normalize_entity(e, "L0") for e in (l0_entities or [])]
    normalized_l1 = [_normalize_entity(e, "L1") for e in (l1_entities or [])]
    normalized_l2 = [_normalize_entity(e, "L2") for e in (l2_entities or [])]
    normalized_l25 = [_normalize_entity(e, "L2.5") for e in (l25_entities or [])]
    normalized_l3 = [_normalize_entity(e, "L3") for e in (l3_entities or [])]

    return normalized_l0, normalized_l1, normalized_l2, normalized_l25, normalized_l3


__all__ = ["normalize_entities", "normalize_status", "_normalize_name", "_camelize_name"]