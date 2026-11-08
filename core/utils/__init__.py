from __future__ import annotations

from .file_helpers import (
    slug_text,
    load_json,
    load_json_optional,
    load_json_cached,
    load_json_required,
    load_json_object,
    append_jsonl,
    resolve_path,
    strip_timestamp_suffix,
    detect_source_profile_from_paths,
    deduplicate_by_key,
    deduplicate_relationships,
)

from .relationship_helpers import (
    RELATIONSHIP_PROVENANCE_VALUES,
    entity_to_dict,
    collect_entity_ids,
    relationship_endpoints,
    is_valid_relationship,
    normalize_relationship_dict,
    prune_dangling_relationships,
)

from .json_parser import (
    ParseResult,
    safe_json_parse,
    safe_json_load,
    extract_json_blocks,
    normalize_text_to_json_candidate,
    repair_common_json_issues,
    parse_llm_output,
    validate_structure,
)

__all__ = [
    # file_helpers
    "slug_text",
    "load_json",
    "load_json_optional",
    "load_json_cached",
    "load_json_required",
    "load_json_object",
    "append_jsonl",
    "resolve_path",
    "strip_timestamp_suffix",
    "detect_source_profile_from_paths",
    "deduplicate_by_key",
    "deduplicate_relationships",
    # relationship_helpers
    "RELATIONSHIP_PROVENANCE_VALUES",
    "entity_to_dict",
    "collect_entity_ids",
    "relationship_endpoints",
    "is_valid_relationship",
    "normalize_relationship_dict",
    "prune_dangling_relationships",
    # json_parser
    "ParseResult",
    "safe_json_parse",
    "safe_json_load",
    "extract_json_blocks",
    "normalize_text_to_json_candidate",
    "repair_common_json_issues",
    "parse_llm_output",
    "validate_structure",
]
