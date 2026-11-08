"""Pipeline validation use case (B3, 2026-08-15).

Extracted from ``MainPipeline._run_contract_validation_chain`` /
``_record_validation_chain_reports`` / ``_validate_pipeline_output`` so the
contract-validation chain and the post-execution output validation are pure,
independently testable steps.

This module owns:
- ``run_contract_validation``: run the full contract-validation chain for a
  pipeline context and return a ``ContractValidationOutcome`` (valid + error
  chips + warning chips). Any failure raises nothing here; the outcome's
  ``valid`` flag is set False with a chip appended.
- ``validate_pipeline_output``: run the output-validation step on assembled
  output data and return a ``PipelineOutputValidation`` outcome with error /
  warning chips.

``MainPipeline`` keeps only the state wiring (append chips to its error /
warning lists, decide whether to stop) and delegates domain logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ContractValidationOutcome:
    """Result of running the pre-execution contract-validation chain."""
    valid: bool
    error_chips: List[str] = field(default_factory=list)
    warning_chips: List[str] = field(default_factory=list)


@dataclass
class PipelineOutputValidation:
    """Result of the post-execution output-validation step."""
    valid: bool
    error_chips: List[str] = field(default_factory=list)
    warning_chips: List[str] = field(default_factory=list)


def run_contract_validation(
    validation_chain: Any,
    *,
    source_path: str,
    domain: str,
    run_id: str,
) -> ContractValidationOutcome:
    """Run the full contract-validation chain.

    ``validation_chain`` is injected (typically ``get_validation_chain()``) so
    the reporter logic is pure and testable without touching global state.
    """
    pipeline_context = {
        "source_path": source_path,
        "domain": domain,
        "run_id": run_id,
    }
    valid, reports = validation_chain.run_full_chain(pipeline_context)

    error_chips: List[str] = []
    warning_chips: List[str] = []
    for report in reports:
        if report.has_errors:
            for error in report.errors:
                error_chips.append(f"Validation {report.validation_id}: {error.message}")
        if report.warnings:
            warning_chips.extend(report.warnings)
    return ContractValidationOutcome(valid=valid, error_chips=error_chips, warning_chips=warning_chips)


def validate_pipeline_output(validation_chain: Any, output_data: Dict[str, Any]) -> PipelineOutputValidation:
    """Validate assembled pipeline output against the output contract."""
    output_report = validation_chain.validate_output(output_data)

    error_chips: List[str] = []
    warning_chips: List[str] = []
    if output_report.has_errors:
        for error in output_report.errors:
            error_chips.append(f"Output validation: {error.message}")
    if output_report.warnings:
        warning_chips.extend(output_report.warnings)
    return PipelineOutputValidation(valid=not error_chips, error_chips=error_chips, warning_chips=warning_chips)