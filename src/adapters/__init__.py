"""Consolidated Adapters - All adapter implementations.

This package contains ALL adapter implementations, consolidating:
- Infrastructure adapters (from src/infrastructure/adapters/)
- Outbound adapters (LLM, artifact stores, parsers, skills)
- Plugin adapters
- Patch adapters

Available adapters:
- DeduplicationAdapter: Implements DeduplicationPort
- QualityAssessmentAdapter: Implements QualityAssessmentPort
- OntologyExtractorAdapter: Implements OntologyExtractorPort
- OntologyPatchAdapter: Implements patch operations

Usage:
    from src.adapters import DeduplicationAdapter

    adapter = DeduplicationAdapter()
    unique_nodes = adapter.deduplicate_nodes(nodes)
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DeduplicationAdapter",
    "QualityAssessmentAdapter",
    "OntologyExtractorAdapter",
    "OntologyPatchAdapter",
]

_MODULE_MAP = {
    "DeduplicationAdapter": ".deduplication_adapter",
    "QualityAssessmentAdapter": ".quality_assessment_adapter",
    "OntologyExtractorAdapter": ".ontology_extractor_adapter",
    "OntologyPatchAdapter": ".ontology_patch_adapter",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_MODULE_MAP[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
