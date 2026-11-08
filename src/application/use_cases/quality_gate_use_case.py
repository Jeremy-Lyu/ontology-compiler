"""Quality-gate use case (B1, 2026-08-15).

Extracted from ``MainPipeline._run_quality_gate`` / ``_build_quality_gate_metadata``
so the governance audit is a pure, independently testable step. This module owns:

- ``run_quality_gate``: execute the governance engine audit on pipeline output
  and return a ``QualityGateOutcome`` (report + fatal/warning chips + metadata).
- ``quality_gate_metadata``: the single ``quality_gate`` metadata dict written
  back into ``context.metadata`` / stage stats.

``MainPipeline`` keeps only the state wiring (append to errors/warnings, write
metadata into context) and delegates domain logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.infrastructure.governance.quality_gate import get_governance_engine


@dataclass
class QualityGateOutcome:
    """Result of running the governance audit on pipeline output."""
    report: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    fatal_errors: List[str] = field(default_factory=list)
    warning_messages: List[str] = field(default_factory=list)


def quality_gate_metadata(output_data: Dict[str, Any], quality_report: Any) -> Dict[str, Any]:
    """Build the ``quality_gate`` metadata block for a governance audit report.

    classification_metrics are carried through from the output metadata (the
    pipeline classification step), never recomputed here.
    """
    classification_metrics = {}
    metadata = output_data.get("metadata") if isinstance(output_data, dict) else {}
    if isinstance(metadata, dict):
        classification_metrics = metadata.get("classification_metrics", {}) or {}

    quality_score = float(getattr(quality_report.score, "overall", 0.0) or 0.0)
    quality_reason = getattr(quality_report.grade, "value", "") or ""
    validation_passed = quality_report.result.value != "fail"

    return {
        "quality_score": quality_score,
        "quality_grade": quality_report.grade.value,
        "critical_issues": quality_report.critical_issue_count,
        "high_issues": quality_report.high_issue_count,
        "classification_metrics": classification_metrics,
        "classification_report": classification_metrics,
        "quality_reason": quality_reason,
        "validation_passed": validation_passed,
    }


def run_quality_gate(output_data: Dict[str, Any]) -> QualityGateOutcome:
    """Run the governance engine audit and package its outcome.

    Returns fatal/warning message chips the caller appends to its error/warning
    lists, plus the metadata block for the run. Pure with respect to the
    caller's state (no writes to pipeline results/stats occur here).
    """
    governance_engine = get_governance_engine()
    quality_report = governance_engine.run_audit(output_data)

    fatal_errors: List[str] = []
    warning_messages: List[str] = []
    if quality_report.result.value == "fail":
        fatal_errors.append(f"Quality gate failed: {quality_report.grade.value}")
    elif quality_report.result.value == "warn":
        warning_messages.append(f"Quality gate warning: {quality_report.grade.value}")

    return QualityGateOutcome(
        report=quality_report,
        metadata=quality_gate_metadata(output_data, quality_report),
        fatal_errors=fatal_errors,
        warning_messages=warning_messages,
    )