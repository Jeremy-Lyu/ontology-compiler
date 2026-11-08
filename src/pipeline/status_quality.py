"""Status/quality derivation extracted from MainPipeline (P0-3 batch 11).

Final status + quality derivation, moved verbatim from MainPipeline methods
using the established ``pipeline``-first-parameter pattern (same as
result_helpers, stage_plumbing, trace_provenance, regression_metadata).

Contents:
- :func:`derive_pipeline_status` — from ``_derive_pipeline_status``
- :func:`extract_stage_quality` — legacy A-2 extract diagnostic view
- :func:`derive_quality_status` — A-2 gate-only quality derivation
- :func:`derive_execution_reason` — from ``_derive_execution_reason``
- :func:`build_pipeline_result` — from ``_build_pipeline_result``

Behavior stays locked by tests/pipeline/test_main_pipeline_characterization.py
(quality-status non-fabrication regressions, per-file extract aggregation,
and the end-to-end run snapshot paths).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover - type hints only, avoids circular import
    from .main_pipeline import (
        MainPipeline,
        PipelineExecutionMode,
        PipelineExecutionResult,
        PipelineQualityStatus,
        PipelineStatus,
    )


def derive_pipeline_status(pipeline: "MainPipeline") -> "PipelineStatus":
    """Derive the coarse pipeline status from stage counters."""
    from .main_pipeline import PipelineStatus

    if pipeline._stats.failed_stages == 0:
        return PipelineStatus.SUCCESS
    if pipeline._stats.completed_stages > 0:
        return PipelineStatus.PARTIAL
    return PipelineStatus.FAILED


def extract_stage_quality(pipeline: "MainPipeline") -> dict[str, Any]:
    """Per-file extract quality view (diagnostics only — not the gate score).

    Kept for diagnostics only — the governance quality gate is the single
    source of truth for the final / master quality score. This view may report
    a value that disagrees with the gate; do not feed it into
    ``derive_quality_status``.
    """
    extract_result = pipeline._stage_results.get("extract")
    if not extract_result or not extract_result.data:
        return {}
    extraction_quality = extract_result.data.get("extraction_quality")
    if not isinstance(extraction_quality, dict):
        return {}
    files = extraction_quality.get("files") or []
    score_fields = []
    file_rows = []
    for item in files:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("quality_score") or 0.0)
        except Exception:
            continue
        score_fields.append(score)
        file_rows.append(
            {
                "file": item.get("file") or item.get("path") or "",
                "quality_score": round(score, 3),
                "validation_passed": bool(item.get("validation_passed")),
                "extraction_quality_status": item.get("extraction_quality_status")
                or item.get("status")
                or "",
            }
        )
    quality_score = sum(score_fields) / len(score_fields) if score_fields else 0.0
    explicit_validation = extraction_quality.get("validation_passed")
    if explicit_validation is not None:
        validation_passed = bool(explicit_validation)
    else:
        validation_passed = bool(score_fields) and all(float(s) > 0.0 for s in score_fields if s is not None) if score_fields else False
    return {
        "validation_passed": validation_passed,
        "quality_score": round(quality_score, 3),
        "extraction_quality_status": extraction_quality.get("status", ""),
        "files": file_rows,
        "file_count": len(file_rows),
        "min_file_score": round(min(score_fields), 3) if score_fields else 0.0,
        "max_file_score": round(max(score_fields), 3) if score_fields else 0.0,
    }


def quality_score_breakdown(pipeline: "MainPipeline") -> dict[str, Any]:
    """Split per-file extract scores from the master/gate score.

    Prevents mistaking a single DEGRADED fragment for the pipeline gold score
    after merge into master.
    """
    file_view = extract_stage_quality(pipeline)
    quality_meta = (
        pipeline.context.metadata.get("quality_gate", {})
        if isinstance(pipeline.context.metadata, dict)
        else {}
    )
    try:
        gate_score = float(quality_meta.get("quality_score", 0.0) or 0.0)
    except Exception:
        gate_score = 0.0
    return {
        "file_avg_score": file_view.get("quality_score", 0.0),
        "file_min_score": file_view.get("min_file_score", 0.0),
        "file_max_score": file_view.get("max_file_score", 0.0),
        "file_count": file_view.get("file_count", 0),
        "files": file_view.get("files") or [],
        "gate_score": round(gate_score, 3),
        "gate_ran": bool(quality_meta),
        "gate_validation_passed": bool(quality_meta.get("validation_passed", False)),
        "note": (
            "file_* scores are extract fragments; gate_score is the only "
            "publish/master quality number"
        ),
    }


def derive_quality_status(
    pipeline: "MainPipeline",
    pipeline_status: "PipelineStatus",
) -> "tuple[PipelineQualityStatus, str, bool, float]":
    """Derive the final quality status.

    [A-2 2026-08-15] 质量评分**唯一来源**为 governance 质量门写入的
    ``context.metadata["quality_gate"]``。此前在 gate 缺失时回落到
    extract stage 的 ``extraction_quality`` 逐文件平均分——这意味着同一
    pipeline 可同时存在两个不同质量结论（如 extract 0.87 / gate fail），
    VSM 超时场景曾因此显示误导性的 ``0.87 silver``。现在 gate 缺失时
    不再伪造分数：有输出 → PARTIAL_SUCCESS(score=0.0, reason=gate not run)，
    无输出 → FAILURE。
    """
    from .main_pipeline import PipelineQualityStatus, PipelineStatus

    quality_meta = pipeline.context.metadata.get("quality_gate", {}) if isinstance(pipeline.context.metadata, dict) else {}
    gate_ran = bool(quality_meta)
    validation_passed = bool(quality_meta.get("validation_passed", False))
    try:
        quality_score = float(quality_meta.get("quality_score", 0.0) or 0.0)
    except Exception:
        quality_score = 0.0
    quality_reason = str(quality_meta.get("quality_reason", "")).strip()
    has_output = bool(pipeline._stats.entity_count or pipeline._stats.relation_count)
    has_errors = bool(pipeline._errors)

    if pipeline_status == PipelineStatus.FAILED:
        if not quality_reason:
            quality_reason = "pipeline failed"
        return PipelineQualityStatus.FAILURE, quality_reason, False, quality_score

    if validation_passed and pipeline_status == PipelineStatus.SUCCESS and quality_score >= 0.6 and not has_errors:
        return PipelineQualityStatus.SUCCESS, quality_reason or "validation and quality thresholds met", True, quality_score

    if has_output and not has_errors:
        if not gate_ran:
            # Gate did not write metadata (e.g. guard/audit stop or no
            # collectable output): never fabricate a score from another
            # source. Output exists, so mark PARTIAL without a score.
            return (
                PipelineQualityStatus.PARTIAL_SUCCESS,
                quality_reason or "quality gate did not run; output produced without gate validation",
                False,
                0.0,
            )
        if pipeline_status == PipelineStatus.SUCCESS and (quality_score >= 0.6 or not validation_passed):
            reason = (
                quality_reason
                or ("output available but quality thresholds not fully met" if quality_score < 0.6 else "output produced in degraded mode; validation did not fully pass")
            )
            return PipelineQualityStatus.DEGRADED_SUCCESS, reason, validation_passed, quality_score
        return PipelineQualityStatus.PARTIAL_SUCCESS, quality_reason or "partial structured output produced", validation_passed, quality_score

    return PipelineQualityStatus.FAILURE, quality_reason or "no publishable structured output", False, quality_score


def derive_execution_reason(pipeline: "MainPipeline") -> str:
    """Derive the human-readable execution-mode reason."""
    from .main_pipeline import PipelineExecutionMode

    execution_reason = pipeline._execution_reason
    if not execution_reason and pipeline._execution_mode != PipelineExecutionMode.PRODUCTION:
        return f"{pipeline._execution_mode.value} mode activated"
    return execution_reason


def build_pipeline_result(
    pipeline: "MainPipeline",
    status: "PipelineStatus",
    execution_reason: str,
    quality_status: "PipelineQualityStatus",
    quality_reason: str,
    validation_passed: bool,
    quality_score: float,
) -> "PipelineExecutionResult":
    """Assemble the ``PipelineExecutionResult`` for the finalized run."""
    from .main_pipeline import (
        PipelineExecutionResult,
        PipelineQualityStatus,
        PipelineStatus,
    )

    if isinstance(pipeline.context.metadata, dict):
        pipeline.context.metadata["quality_breakdown"] = quality_score_breakdown(pipeline)

    return PipelineExecutionResult(
        run_id=pipeline._run_id,
        status=status,
        context=pipeline.context,
        stats=pipeline._stats,
        stage_results=pipeline._stage_results,
        output_files=pipeline._output_files,
        errors=pipeline._errors,
        warnings=pipeline._warnings,
        publishable=(status == PipelineStatus.SUCCESS and not pipeline._errors and quality_status == PipelineQualityStatus.SUCCESS),
        execution_mode=pipeline._execution_mode.value,
        execution_reason=execution_reason,
        quality_status=quality_status,
        quality_reason=quality_reason,
        validation_passed=validation_passed,
        quality_score=quality_score,
    )
