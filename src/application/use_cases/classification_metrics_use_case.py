"""Classification-metrics use case (B2, 2026-08-15).

Extracted from ``MainPipeline._collect_classification_metrics`` /
``_summarize_classification_entities`` / ``_collect_classification_entities``
so classification summarization is a pure, independently testable step.

This module owns:
- ``collect_classification_entities``: gather entity/instance dicts from the
  list-shaped ``nodes`` / ``instances`` payloads of stage results.
- ``summarize_classification``: run the classification service's ``summarize``
  over those entities and return ``summary.to_dict()`` (or None when empty).

``MainPipeline`` keeps only the state wiring (write back to
``context.metadata["classification_metrics"]`` / stats). Domain logic lives here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def collect_classification_entities(stage_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect entity dicts across stage results for classification metrics.

    Only list-shaped ``nodes`` / ``instances`` payloads are considered; other
    stage keys (relationships, metadata, etc.) are ignored, matching the
    previous in-place behavior byte-for-byte.
    """
    classification_entities: List[Dict[str, Any]] = []
    for stage_result in stage_results.values():
        if not stage_result or not getattr(stage_result, "data", None):
            continue
        stage_data = stage_result.data
        for key in ("nodes", "instances"):
            value = stage_data.get(key)
            if isinstance(value, list):
                classification_entities.extend([item for item in value if isinstance(item, dict)])
    return classification_entities


def summarize_classification(
    stage_results: Dict[str, Any],
    *,
    get_classification_service: Any,
) -> Optional[Dict[str, Any]]:
    """Summarize classification metrics over stage entity payloads.

    Returns ``summary.to_dict()``, or ``None`` when there are no entity
    payloads to summarize (the caller then skips writing metrics).
    """
    entities = collect_classification_entities(stage_results)
    if not entities:
        return None
    classification_service = get_classification_service()
    summary = classification_service.summarize(entities)
    if summary is None:
        return None
    to_dict = getattr(summary, "to_dict", None)
    return to_dict() if to_dict else None