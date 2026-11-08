"""Presentation-layer renderer re-exports (B-5, 2026-08-17).

The 18 pure renderer functions were split into report families under
``src.application.rag.presentation`` (G10). This module keeps the old flat
import surface working::

    from src.application.rag.formatting import format_neo4j_result

continues to work; new code should import from ``presentation`` directly.
"""

from __future__ import annotations

from .presentation.neo4j_results import (  # noqa: F401
    format_neo4j_result,
    format_document_result,
)
from .presentation.dify_structured import (  # noqa: F401
    format_dify_structured_answer,
    try_parse_dify_json,
)
from .presentation.ah_framework import (  # noqa: F401
    identify_target_entity,
    collect_related_entities,
    categorize_relation,
    extract_business_context,
    generate_ah_framework_report,
    build_section_d,
    build_section_e,
    build_section_f,
)
from .presentation.natural_language import (  # noqa: F401
    synthesize_natural_answer,
    generate_natural_language_report,
)
from .presentation.sections_a_f import (  # noqa: F401
    build_section_a,
    build_section_b,
    build_section_c,
    compute_difference_hint,
)

__all__ = [
    "format_neo4j_result",
    "format_document_result",
    "format_dify_structured_answer",
    "build_section_a",
    "build_section_b",
    "build_section_c",
    "build_section_d",
    "build_section_e",
    "build_section_f",
    "compute_difference_hint",
    "generate_natural_language_report",
    "try_parse_dify_json",
    "identify_target_entity",
    "categorize_relation",
    "collect_related_entities",
    "extract_business_context",
    "generate_ah_framework_report",
    "synthesize_natural_answer",
]