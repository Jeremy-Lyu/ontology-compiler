"""LLM observability aggregation use case (D-4, 2026-08-15).

Extracted so every run's manifest can carry the LLM monitor's aggregated stats
(calls / tokens / cost / success rate / per-stage breakdown) for replay and
cost auditing — the "D-4 observability aggregation" item of the pipeline
stability refactor.

Pure with respect to the caller: ``collect_llm_stats`` takes the monitor
(summary is pulled via its ``get_summary()``) and returns a stable dict, and
``reset_llm_monitor`` resets the monitor between runs to avoid cross-run bleed.
"""

from __future__ import annotations

from typing import Any, Dict


def collect_llm_stats(monitor: Any) -> Dict[str, Any]:
    """Aggregate the LLM monitor summary into a stable manifest block.

    Falls back to an empty stats dict when the monitor is unavailable or has no
    enabled aggregation, so the manifest writer never depends on monitoring
    internals.
    """
    if monitor is None:
        return {"total_calls": 0, "total_tokens": 0, "total_cost_estimate": 0.0, "stage_metrics": {}}
    try:
        if hasattr(monitor, "get_summary"):
            summary = monitor.get_summary() or {}
            if isinstance(summary, dict):
                return summary
    except Exception:
        pass
    return {"total_calls": 0, "total_tokens": 0, "total_cost_estimate": 0.0, "stage_metrics": {}}


def reset_llm_monitor(monitor: Any) -> None:
    """Reset the monitor so a new run starts with clean counters.

    Deliberately no-op when the monitor lacks ``reset`` (test fakes / disabled
    monitors), so this never breaks the run lifecycle.
    """
    if monitor is None:
        return
    try:
        reset = getattr(monitor, "reset", None)
        if reset is not None:
            reset()
    except Exception:
        pass