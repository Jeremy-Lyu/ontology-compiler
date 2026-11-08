"""Failure-memory strategy adjustment extracted from TrueUnifiedExtractor.

Phase 3 LOC reduction (docs/CODE_CLEANUP_CANDIDATES.md §3.1): the
[P3-EXP-3] failure-memory strategy check moved verbatim into a pure helper.
The extractor calls :func:`apply_failure_memory_adjustments` before running
extraction; behavior is identical, including silent degradation when the
failure memory service is unavailable.

All functions take explicit inputs (no extractor state), matching the
pipeline-first/explicit-input pattern used by result_helpers and
stage_execution_core in src/pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from ontology_compiler.services.failure_memory import (
        get_failure_memory,
        record_extraction_failure,
        record_schema_violation,
    )
except ImportError:  # pragma: no cover - service optional at runtime
    def get_failure_memory(): return None
    def record_extraction_failure(*args, **kwargs): pass
    def record_schema_violation(*args, **kwargs): pass


def apply_failure_memory_adjustments(
    document_key: str,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """Consult failure memory and return extraction adjustments for *document_key*.

    Returns a dict::

        {"fast_mode": bool}

    Mutates ``ctx["prompt_strategy"]`` in place when the memory recommends a
    different prompt strategy. Never raises — any internal failure logs a
    warning and returns defaults (same semantics as the original inline block
    in ``TrueUnifiedExtractor.extract``).
    """
    fast_mode = bool(ctx.get("fast_mode", False))
    try:
        memory = get_failure_memory()
        if memory is None:
            adjustment = {"should_adjust": False, "recent_failure_count": 0, "recommended_adjustments": []}
        else:
            adjustment = memory.check_should_adjust_strategy(document_key)
        if adjustment["should_adjust"]:
            logger.warning(
                f"Failure memory detected {adjustment['recent_failure_count']} recent failures "
                f"for document '{document_key}'. Applying adjustments: "
                f"{[a['type'] for a in adjustment['recommended_adjustments']]}."
            )
            # Apply strategy adjustment if recommended
            adjusted_strategy = memory.get_adjusted_strategy(
                document_key, ctx.get("prompt_strategy", "general")
            )
            if adjusted_strategy != ctx.get("prompt_strategy", "general"):
                ctx["prompt_strategy"] = adjusted_strategy
                logger.info(f"Auto-switched prompt strategy to '{adjusted_strategy}' based on failure memory.")
            # Enable fast_mode if extraction failures detected
            for adj in adjustment["recommended_adjustments"]:
                if adj["type"] == "fallback_mode":
                    fast_mode = True
                    logger.info("Auto-enabled fast_mode due to recurring extraction failures.")
                    break
    except Exception as e:
        logger.warning(f"Failure memory strategy check failed: {e}")
    return {"fast_mode": fast_mode}
