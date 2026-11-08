"""Application service compatibility exports.

This package is intentionally thin. Import concrete services from their
own modules, and import feedback primitives from `src.application.feedback`.
"""
from __future__ import annotations

from src.application.services.feedback_collector import FeedbackCollector, FeedbackRecord, get_global_collector
from src.application.services.feedback_loop import FeedbackLoop, OptimizationType

# Contract-aware builders. Migration target: src.domain.models.core (ContractModel
# subclasses) + src.application.bootstrap.llm_gateway for trace helpers. The
# dedicated contract_service module was never created; fallback shims below keep
# imports stable with lazy AttributeError on call for any missing symbol.
_CONTRACT_ALL: list[str] = [
    "build_trace_context",
    "build_artifact_envelope",
    "build_retrieval_asset_packet",
    "build_skill_definition",
    "build_pipeline_stage_result",
    "build_pipeline_execution_result",
    "build_validation_check",
    "build_validation_report",
]

try:
    from src.application.services.contract_service import (  # type: ignore[attr-defined]
        build_artifact_envelope,
        build_pipeline_execution_result,
        build_pipeline_stage_result,
        build_retrieval_asset_packet,
        build_skill_definition,
        build_trace_context,
        build_validation_check,
        build_validation_report,
    )
except ImportError:
    import logging as _logging
    import typing as _t

    _logger = _logging.getLogger(__name__)
    _logger.debug(
        "src.application.services.contract_service not present; compatibility "
        "stubs installed for build_* symbols (importable, raise AttributeError at call).",
    )

    def _make_stub(name: str) -> _t.Any:
        def _stub(*args: _t.Any, **kwargs: _t.Any) -> _t.Any:  # noqa: D401
            raise AttributeError(
                f"{name}() is not implemented. The contract_service module was never "
                "created; please migrate callers to src.domain.models.core ContractModel "
                "helpers or construct the payload inline.",
            )
        _stub.__name__ = name
        _stub.__qualname__ = name
        return _stub

    for _sym in _CONTRACT_ALL:
        locals()[_sym] = _make_stub(_sym)


__all__ = [
    "FeedbackCollector",
    "FeedbackRecord",
    "get_global_collector",
    "FeedbackLoop",
    "OptimizationType",
    *_CONTRACT_ALL,
]
