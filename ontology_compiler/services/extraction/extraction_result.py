"""Data models for the unified extraction pipeline.

``UnifiedExtractionResult`` is the shared container produced by the extractor
(both the single-shot path and the chunked path) and consumed by the
post-processing pipeline. It lives in its own module so the chunked-extraction
and validation services can construct it without creating a circular import
back into ``true_unified_extractor``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class UnifiedExtractionResult:
    """Result from unified extraction containing all five layers + relationships."""
    l0_entities: List[Dict[str, Any]]
    l1_entities: List[Dict[str, Any]]
    l2_entities: List[Dict[str, Any]]
    l25_entities: List[Dict[str, Any]]
    l3_entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    raw_response: str
    processing_time_ms: float
    token_usage: Dict[str, int]
    quality_score: float = 0.0
    degraded: bool = False  # Phase 1: Flag for degraded mode results
